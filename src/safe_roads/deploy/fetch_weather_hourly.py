from datetime import datetime, timedelta
import logging
import pandas as pd
from meteostat import Hourly, Point
from sqlalchemy import text, create_engine

from safe_roads.utils.config import get_pg_url
from safe_roads.utils.data import df_to_pg, _table_exists

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


def write_to_postgis(df: pd.DataFrame, db_url: str, table_name: str) :

    if df is None or df.empty:
        return  # nothing to do

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False,
        )

def get_hourly_weather() -> None:
    logger.info("Starting current hourly weather ingestion for Greater London")

    db_url = get_pg_url()
    df = fetch_hourly_weather(LAT, LON)
    write_to_postgis(df, db_url, table_name=TABLE_NAME)

    logger.info("Weather ingestion complete")


if __name__ == "__main__":
    get_hourly_weather()
