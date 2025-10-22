from sqlalchemy import create_engine, text
from pathlib import Path
from prefect import task, flow, get_run_logger
import sqlparse

from safe_roads.utils.config import get_pg_url
from safe_roads.utils.data import _run_sql_file, _table_exists

@task(name="Standardise Column Names")
def rename_column(db_url: str, sql_path: Path):
    logger = get_run_logger()

    sql_path = Path(sql_path)
    token = sql_path.stem.split('_')[1]

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    logger.info(f"Running SQL from {sql_path}")
    _run_sql_file(db_url, sql_path)
    logger.info(f"Renamed columns to {token} in all matching tables.")


@task(name="Merge All Columns")
def merge_all_columns(db_url: str, sql_path: Path):
    logger = get_run_logger()

    sql_path = Path(sql_path)

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    logger.info(f"Running SQL from {sql_path}")
    _run_sql_file(db_url, sql_path)
    logger.info(f"Merged all collision data tables")


@task(name='Prepare Collision Data')
def prepare_collision_data():
    url = get_pg_url()
    logger = get_run_logger()
    

    logger.info("Standardising Time Column")
    rename_column(url, sql_path='src\safe_roads\query\standardize_time_column.sql')
    
    logger.info("Standardising collisonID Column")
    rename_column(url, sql_path='src\safe_roads\query\standardize_collisionID_column.sql')

    logger.info("Standardising date Column")
    rename_column(url, sql_path='src\safe_roads\query\standardize_date_column.sql')

    logger.info("Standardising Borough Column")
    rename_column(url, sql_path='src\safe_roads\query\standardize_borough_column.sql')

    logger.info("Merging all collision data")
    merge_all_columns(url, sql_path='src\safe_roads\query\merge_all_attendant_tables.sql')
    
    logger.info("Standardising Time Data")
    _run_sql_file(url, sql_path='src\safe_roads\query\change_time_data.sql')
    logger.info("time_parsed column created")

    logger.info("Standardising Date Data")
    _run_sql_file(url, sql_path='src\safe_roads\query\change_date_data.sql')
    logger.info("date_parsed column created")

    logger.info("Creating datetime column")
    _run_sql_file(url, sql_path='src\safe_roads\query\create_date_time_column.sql')
    logger.info("date_parsed column created")

    logger.info("collisiondata table merged and standardised")


if __name__ == '__main__':
    prepare_collision_data()