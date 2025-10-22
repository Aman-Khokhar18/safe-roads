from prefect import task, get_run_logger
from pathlib import Path
from safe_roads.utils.data import _run_sql_file, _table_exists
from safe_roads.utils.config import get_pg_url

@task(name="Transform Data")
def transform_data():
    logger = get_run_logger()
    url = get_pg_url()
    clean_osm_enriched = Path("src/safe_roads/query/deploy/clean_osm_data_live.sql")

    logger.info("Running SQL Queries")
    if _table_exists(url=url, table="osm_data_live_h3_enriched"):
        _run_sql_file(clean_osm_enriched)
    else:
        raise FileNotFoundError("osm_data_live_h3_enriched table not found in database")