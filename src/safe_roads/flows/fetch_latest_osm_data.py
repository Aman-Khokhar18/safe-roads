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
def download_latest_osm_road_data(gdf: gpd.GeoDataFrame):
    """Fetch the latest available OSM elements (highway=*) from ohsome."""
    logger = get_run_logger()
    client = OhsomeClient()

    # latest available timestamp from server, e.g. '2025-10-20T00:00:00Z'
    latest_ts = client.end_timestamp
    latest_date = pd.to_datetime(latest_ts, utc=True)
    year = str(latest_date.year)
    month = f"{latest_date.month:02d}"

    logger.info(f"Requesting OSM road data at latest snapshot: {latest_ts}")

    # omit `time` -> API returns latest snapshot by default
    res = client.elements.geometry.post(
        bpolys=gdf,
        filter="highway=* and (type: node or type:way)",
        properties="tags,metadata"
    )
    df = res.as_dataframe()
    df["year"] = year
    df["month"] = month

    logger.info(f"Downloaded {len(df)} elements for latest snapshot {year}-{month}")
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
        "overtaking", "bridge", "tunnel", "layer", "incline", "barrier", "amenity"
    ]

    other = pd.DataFrame.from_records(df["@other_tags"], index=df.index)
    out = pd.DataFrame(index=df.index)
    for k in tags:
        out[k] = df[k] if k in df.columns else other.get(k)

    if "year" in df.columns:  out["year"] = df["year"]
    if "month" in df.columns: out["month"] = df["month"]

    crs = df.crs if isinstance(df, gpd.GeoDataFrame) else None
    result = gpd.GeoDataFrame(pd.concat([out, df[["geometry"]]], axis=1), geometry="geometry", crs=crs)

    logger.info(f"Formatted {len(result)} rows")
    return result


@task
def load_osm_data_to_sql(db_url: str, gdf: gpd.GeoDataFrame):
    logger = get_run_logger()
    logger.info(f"Loading {len(gdf)} rows into PostGIS table public.OSM_data")

    engine = create_engine(db_url)
    geom_dtype = {"geometry": Geometry(geometry_type="GEOMETRY", srid=4326)}

    gdf.to_postgis(name="osm_data_latest", con=engine, schema="public", if_exists="replace", dtype=geom_dtype)
    
    logger.info("Load complete")


@flow(name="Ingest latest OSM data to PostGIS")
def ingest_latest_osm_data():
    logger = get_run_logger()
    logger.info("Starting latest OSM ingestion flow")

    _ = load_config()  
    url = get_pg_url()
    gdf = get_greater_london_gdf()

    osm_df = download_latest_osm_road_data(gdf)
    osm_df = format_columns(osm_df)
    load_osm_data_to_sql(url, osm_df)

    logger.info("Latest snapshot processed successfully")


if __name__ == "__main__":
    ingest_latest_osm_data()
