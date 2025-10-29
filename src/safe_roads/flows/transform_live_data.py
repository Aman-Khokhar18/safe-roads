from prefect import flow, get_run_logger

from safe_roads.utils.config import get_pg_url, load_config
from safe_roads.utils.data import _run_sql_file, _table_exists
from safe_roads.flows.fetch_live_data import fetch_once, fetch_monthly, fetch_hourly

url = get_pg_url()

@flow
def transform_once():
    logger = get_run_logger()
    if _table_exists(url=url, table="london_h3"):
        _run_sql_file("src\safe_roads\query\deploy\london_h3_expand.sql")
    else: 
        logger.info("london_h3 table not found")
        fetch_once()

@flow
def transform_monthly():
    logger = get_run_logger()
    if _table_exists(url=url, table="osm_data_live"):
        _run_sql_file("src\safe_roads\query\deploy\osm_data_h3_convert.sql")
        _run_sql_file("src\safe_roads\query\deploy\osm_data_live_enriched.sql")
    else:
        logger.info("osm_live_data is not found")
        fetch_monthly()


@flow 
def transform_hourly():
    logger = get_run_logger()
    if _table_exists(url=url, table="osm_data_live_h3_enriched"):
        _run_sql_file("src\safe_roads\query\deploy\clean_osm_data_live.sql")
    else:
        fetch_hourly()