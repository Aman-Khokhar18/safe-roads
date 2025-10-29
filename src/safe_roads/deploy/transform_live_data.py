import logging

from safe_roads.utils.data import _table_exists
from safe_roads.query.deploy import run_sql
from safe_roads.utils.config import get_pg_url

logger = logging.getLogger("safe_roads.transform")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    )


def transform_data() -> None:
    url = get_pg_url()

    logger.info("Running SQL queries")
    if _table_exists(url=url, table="osm_deploy_latest"):
        run_sql(url, "add_weather_live.sql")
        logger.info("Added live weather data")

        run_sql(url, "cyclic_time_features.sql")
        logger.info("Created time features")
    else:
        msg = "OSM latest data table not found in database"
        logger.error(msg)
        raise FileNotFoundError(msg)


if __name__ == "__main__":
    transform_data()
