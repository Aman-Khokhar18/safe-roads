from prefect import task, get_run_logger
from pathlib import Path
from safe_roads.utils.data import _table_exists
from safe_roads.query.deploy import run_sql
from safe_roads.utils.config import get_pg_url

@task(name="Transform Data")
def transform_data():
    logger = get_run_logger()
    url = get_pg_url()

    logger.info("Running SQL Queries")
    if _table_exists(url=url, table="osm_data_live_h3_enriched"):
        run_sql(url, "clean_osm_data_live.sql")
    else:
        raise FileNotFoundError("osm_data_live_h3_enriched table not found in database")
    
if __name__ == "__main__":
    transform_data()