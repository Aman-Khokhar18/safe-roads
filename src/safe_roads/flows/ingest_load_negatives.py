from prefect import flow, get_run_logger

from safe_roads.utils.config import get_pg_url, load_config
from safe_roads.utils.data import _run_sql_file, _table_exists, any_attendant_tables

from safe_roads.tasks.get_london_h3_data import load_borough_h3_data



@flow(name='Ingest and Load Negative London Data')
def create_negatives():
    url = get_pg_url()
    logger = get_run_logger()
    config = load_config()
    start = config['START_DATE']
    end = config['END_DATE']

    if _table_exists(url, 'london_osm_weather'):
        logger.info("London OSM Weather data table exists")
        return False
    
    elif _table_exists(url, 'london_enriched_daily_sample'):
        logger.info("london_osm_weather table exists")
        logger.info("Creating date time columns")
        _run_sql_file(url, sql_path='src\safe_roads\query\create_london_osm_weather.sql')
        logger.info("Weather table created")
        return True    
    
    elif _table_exists(url, 'london_osm_enriched_daily'):
        logger.info("london_osm_daily table exists")
        logger.info("Creating date time columns")
        _run_sql_file(url, sql_path='src\safe_roads\query\down_sample.sql')
        logger.info("london_osm_h3_enriched table created")
        return True
    
    elif _table_exists(url, 'london_osm_h3_enriched'):
        logger.info("london_osm_h3_enriched table exists")
        logger.info("Creating Time column")
        _run_sql_file(url, sql_path='src\safe_roads\query\create_london_osm_daily.sql')
        logger.info("london_osm_h3_enriched table created")
        return True
    
    elif _table_exists(url, 'london_osm_h3'):
        logger.info("london_osm_h3 table exists")
        logger.info("Creating Geo features")
        _run_sql_file(url, sql_path='src\safe_roads\query\create_london_osm_enriched.sql')
        logger.info("london_osm_h3_enriched table created")
        return True
    
    elif _table_exists(url, 'london_h3'):
        logger.info("london_h3 table exists")
        logger.info("Joining OSM data")
        _run_sql_file(url, sql_path='src\safe_roads\query\create_london_osm_h3.sql')
        logger.info("london_osm_h3 table created")
        return True    
    
    else:
        logger.info("london_h3 does not exists")
        logger.info("Downloading London Borough and H3 Data")
        load_borough_h3_data()
        _run_sql_file(url, sql_path='src\safe_roads\query\london_h3_expand.sql')       

if __name__ == '__main__':
    create_negatives()