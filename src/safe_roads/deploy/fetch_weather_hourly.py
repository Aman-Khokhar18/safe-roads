from datetime import datetime, timezone, timedelta
import pandas as pd
from meteostat import Hourly, Point
from prefect import task, get_run_logger

from safe_roads.utils.config import get_pg_url
from safe_roads.utils.data import df_to_pg

LAT = 51.5074
LON = -0.1278
TIMEZONE = "Europe/London"
TABLE_NAME = "weather_live"


@task(retries=3, retry_delay_seconds=10)
def fetch_hourly_weather(lat: float, lon: float, area_label: str = "Greater London") -> pd.DataFrame:
    log = get_run_logger()
    loc = Point(lat, lon)

    end_utc   = datetime.utcnow()                       
    start_utc = end_utc - timedelta(hours=2)            

    log.info(
        f"Fetching Meteostat Hourly for {area_label} ({lat:.4f}, {lon:.4f}) "
        f"{start_utc.isoformat()} → {end_utc.isoformat()} (timezone='Europe/London')"
    )

    df = Hourly(loc, start_utc, end_utc, timezone="Europe/London").fetch()
    if df is None or df.empty:
        log.warning("No data returned from Meteostat.")
        return pd.DataFrame()

    # Keep only the most recent hour
    df = df.reset_index().rename(columns={"time": "weather_datetime"}).tail(1)
    df["latitude"] = lat
    df["longitude"] = lon
    df["area"] = area_label
    df["retrieved_at_utc"] = pd.Timestamp.utcnow()

    log.info("Fetched latest hourly record")
    return df


@task
def write_to_postgis(df: pd.DataFrame, db_url: str, table_name: str = TABLE_NAME, if_exists: str = "replace"):
    log = get_run_logger()
    if df is None or df.empty:
        log.warning("No data to load; skipping write.")
        return
    df_to_pg(df, table_name, db_url, if_exists=if_exists)
    log.info(f"Loaded {len(df)} rows into {table_name} (if_exists='{if_exists}').")


@task(name="Get Current Hourly Weather for Greater London (Point → PostGIS)")
def get_hourly_weather():
    log = get_run_logger()
    log.info("Starting current hourly weather ingestion for Greater London")

    db_url = get_pg_url()

    df = fetch_hourly_weather(LAT, LON)
    write_to_postgis(df, db_url, table_name=TABLE_NAME, if_exists="replace")

    log.info("Weather ingestion complete")


if __name__ == "__main__":
    get_hourly_weather()
