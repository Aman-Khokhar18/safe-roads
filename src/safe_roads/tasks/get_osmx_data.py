from prefect import task, flow, get_run_logger
import osmnx as ox
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry

from safe_roads.utils.config import get_pg_url
from safe_roads.flows.ingest_OSM_data import get_greater_london_gdf  


@task
def get_all_osm_features_within_london(london_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    logger = get_run_logger()
    logger.info("Downloading all OSM features within Greater London…")

    boundary = london_gdf.to_crs(4326).geometry.unary_union  

    gdf = ox.features_from_polygon(boundary, tags={"highway": True})

    gdf = gdf.to_crs(4326).reset_index(drop=False)  
    logger.info(f"Fetched {len(gdf)} features from OSM")
    return gdf


@task
def save_to_postgis(db_url: str, gdf: gpd.GeoDataFrame, table: str = "osmx_data"):
    logger = get_run_logger()
    engine = create_engine(db_url)
    geom_dtype = {"geometry": Geometry(geometry_type="GEOMETRY", srid=4326)}

    logger.info(f"Writing {len(gdf)} rows to {table}")

    gdf.to_postgis(name=table, con=engine, schema="public",
                   if_exists='replace', dtype=geom_dtype, index=False)
    logger.info("PostGIS write complete")


@flow(name="Save all OSM features inside Greater London (OSMnx → PostGIS)")
def save_london_osmx_data():
    logger = get_run_logger()
    logger.info("Starting OSMnx → PostGIS export")

    db_url = get_pg_url()
    london_gdf = get_greater_london_gdf()
    all_features = get_all_osm_features_within_london(london_gdf)
    save_to_postgis(db_url, all_features, table="osmx_data")

    logger.info("Done")


if __name__ == "__main__":
    save_london_osmx_data()
