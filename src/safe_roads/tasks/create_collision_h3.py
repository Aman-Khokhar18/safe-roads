from prefect import task, get_run_logger
from sqlalchemy import text, create_engine
from pathlib import Path

from safe_roads.utils.data import _run_sql_file
from safe_roads.utils.config import get_pg_url

def create_h3_columns():
    logger = get_run_logger()
    url = get_pg_url()
    sql_path = "src\safe_roads\query\collision_convert_uk_to_h3.sql"
    
    if not Path(sql_path).exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    logger.info(f"Running SQL from {sql_path}")
    _run_sql_file(url, sql_path)

    logger.info("H3_data table created")



