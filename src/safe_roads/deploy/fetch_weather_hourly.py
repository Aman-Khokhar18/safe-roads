from datetime import datetime, timedelta
import logging
import pandas as pd
from meteostat import Hourly, Point

from safe_roads.utils.config import get_pg_url
from safe_roads.utils.data import df_to_pg

logger = logging.getLogger("safe_roads.weather")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    )

LAT = 51.5074
LON = -0.1278
TIMEZONE = "Europe/London"
TABLE_NAME = "weather_live"


def fetch_hourly_weather(lat: float, lon: float, area_label: str = "Greater London") -> pd.DataFrame:
    """Fetch latest hourly record from Meteostat for a point."""
    loc = Point(lat, lon)

    end_utc = datetime.utcnow()
    start_utc = end_utc - timedelta(hours=2)

    logger.info(
        "Fetching Meteostat Hourly for %s (%.4f, %.4f) %s → %s (timezone='%s')",
        area_label, lat, lon, start_utc.isoformat(), end_utc.isoformat(), TIMEZONE
    )

    try:
        df = Hourly(loc, start_utc, end_utc, timezone=TIMEZONE).fetch()
    except Exception as e:
        logger.exception("Meteostat fetch failed: %s", e)
        return pd.DataFrame()

    if df is None or df.empty:
        logger.warning("No data returned from Meteostat.")
        return pd.DataFrame()

    # Keep only the most recent hour
    df = df.reset_index().rename(columns={"time": "weather_datetime"}).tail(1)
    df["latitude"] = lat
    df["longitude"] = lon
    df["area"] = area_label
    df["retrieved_at_utc"] = pd.Timestamp.utcnow()

    logger.info("Fetched latest hourly record")
    return df


def write_to_postgis(df: pd.DataFrame, db_url: str, table_name: str = TABLE_NAME, if_exists: str = "replace") -> None:
    """Write dataframe to PostGIS."""
    if df is None or df.empty:
        logger.warning("No data to load; skipping write.")
        return

    try:
        df_to_pg(df, table_name, db_url, if_exists=if_exists)
        logger.info("Loaded %d rows into %s (if_exists='%s').", len(df), table_name, if_exists)
    except Exception as e:
        logger.exception("Failed to write to PostGIS: %s", e)


def get_hourly_weather() -> None:
    """End-to-end: fetch and write the latest hourly weather for Greater London."""
    logger.info("Starting current hourly weather ingestion for Greater London")

    db_url = get_pg_url()
    df = fetch_hourly_weather(LAT, LON)
    write_to_postgis(df, db_url, table_name=TABLE_NAME, if_exists="replace")

    logger.info("Weather ingestion complete")


if __name__ == "__main__":
    get_hourly_weather()
