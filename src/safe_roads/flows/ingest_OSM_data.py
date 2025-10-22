from prefect import task, flow, get_run_logger
from ohsome import OhsomeClient
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from shapely.ops import unary_union
import osmnx as ox
from tqdm import tqdm


from safe_roads.utils.data import _table_exists
from safe_roads.utils.config import get_pg_url, load_config, year_month



@task
def get_greater_london_gdf():
    logger = get_run_logger()
    logger.info("Geocoding Greater London boundary")
    
    gdf = ox.geocode_to_gdf("Greater London, United Kingdom")
    gdf = gdf.to_crs(4326)    
    geom = unary_union(gdf.geometry.buffer(0))
    london_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    logger.info("Greater London geometry prepared")
    return london_gdf

@task(retries=3, retry_delay_seconds=10)
def download_historical_osm_road_data(gdf: gpd.GeoDataFrame, year: str, month: str):
    logger = get_run_logger()
    logger.info(f"Requesting OSM road data for {year}-{month}")

    client = OhsomeClient()
    res = client.elements.geometry.post(
        bpolys=gdf,
        time=year + '-' + month,
        filter="highway=* and (type: node or type:way)",
        properties="tags,metadata"
    )
    df = res.as_dataframe()
    df['year'] = year
    df['month'] = month
    logger.info(f"Downloaded {len(df)} elements for {year}-{month}")
    return df

@task
def format_columns(df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Formatting columns")

    tags = [
    "highway", "name", "lanes", "width", "surface", "smoothness", "oneway",
    "junction", "maxspeed", "traffic_signals", "traffic_calming",
    "crossing", "sidewalk", "cycleway", "bicycle", "lit",
    "turn:lanes", "access", "vehicle", "hgv", "psv", "bus",
    "overtaking", "bridge", "tunnel", "layer", "incline", "barrier","amenity"
    ]

    other = pd.DataFrame.from_records(df['@other_tags'], index=df.index)
    out = pd.DataFrame(index=df.index)
    for k in tags:
        out[k] = df[k] if k in df.columns else other.get(k)

    if 'year' in df.columns:  out['year'] = df['year']
    if 'month' in df.columns: out['month'] = df['month']

    crs = df.crs if isinstance(df, gpd.GeoDataFrame) else None
    result = gpd.GeoDataFrame(pd.concat([out, df[['geometry']]], axis=1), geometry='geometry', crs=crs)

    logger.info(f"Formatted {len(result)} rows")
    return result

@task
def load_osm_data_to_sql(db_url: str, gdf: gpd.GeoDataFrame):
    logger = get_run_logger()
    logger.info(f"Loading {len(gdf)} rows into PostGIS table OSM_data")

    engine = create_engine(db_url)
    geom_dtype = {"geometry": Geometry(geometry_type="GEOMETRY", srid=4326)}

    if _table_exists:
        gdf.to_postgis(name='OSM_data', con=engine, schema='public', if_exists='append', dtype=geom_dtype)
    else:
        gdf.to_postgis(name='OSM_data', con=engine, schema='public', if_exists='replace', dtype=geom_dtype)
    logger.info("Load complete")
        

@flow(name="Ingesting OSM data to PostGIS")
def ingest_raw_osm_data(monthly: bool = False):
    logger = get_run_logger()
    logger.info("Starting OSM ingestion flow")

    config = load_config()
    url = get_pg_url()
    gdf = get_greater_london_gdf()

    start_date = config['START_DATE']
    end_date = config['END_DATE']

    start_year, start_month = year_month(start_date)
    end_year, end_month = year_month(end_date)
    
    if monthly:
        periods = [(y, m) for y in range(start_year, end_year+1) for m in range(start_month, end_month+1)]
        logger.info(f"Total snapshots: {len(periods)}")
        for year, month in tqdm(periods, desc="Snapshots", unit="month"):
            logger.info(f"Processing snapshot {year}-{month:02d}")
            osm_df = download_historical_osm_road_data(gdf, str(year), f"{month:02d}")
            osm_df = format_columns(osm_df)
            load_osm_data_to_sql(url, osm_df)

    else:
        m = 1
        periods = [(y, m) for y in range(start_year, end_year+1)]
        logger.info(f"Total snapshots: {len(periods)}")
        for year, month in tqdm(periods, desc="Snapshots", unit="month"):
            logger.info(f"Processing snapshot {year}-{month:02d}")
            osm_df = download_historical_osm_road_data(gdf, str(year), f"{month:02d}")
            osm_df = format_columns(osm_df)
            load_osm_data_to_sql(url, osm_df)

    logger.info("All snapshots processed successfully")

if __name__ == "__main__":
    ingest_raw_osm_data()
