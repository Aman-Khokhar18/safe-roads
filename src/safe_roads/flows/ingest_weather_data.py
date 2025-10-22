from datetime import datetime
import pandas as pd
from meteostat import Hourly, Stations
from prefect import flow, get_run_logger

from safe_roads.utils.config import get_pg_url, load_config
from safe_roads.utils.data import df_to_pg


def london_stations(limit: int = 8) -> pd.DataFrame:
    return Stations().nearby(51.5074, -0.1278).fetch(limit).reset_index()


def fetch_station_hourly(station_id, start, end, timezone) -> pd.DataFrame:
    df = Hourly(station_id, start, end, timezone=timezone).fetch()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={'time': 'datetime'})
    df['station_id'] = station_id
    return df


def best_per_timestamp(all_rows: pd.DataFrame, station_order: list[str]) -> pd.DataFrame:
    prio = {sid: i for i, sid in enumerate(station_order)}
    all_rows['station_rank'] = all_rows['station_id'].map(prio).fillna(len(prio)).astype(int)

    exclude = {'datetime', 'station_id', 'station_rank'}
    value_cols = [c for c in all_rows.columns if c not in exclude]

    # fewest NULLs wins; tie -> lower station_rank (nearer) wins
    all_rows['_nulls'] = all_rows[value_cols].isna().sum(axis=1)
    all_rows.sort_values(['datetime', '_nulls', 'station_rank'], kind='mergesort', inplace=True)

    best = all_rows.drop_duplicates(subset=['datetime'], keep='first')
    return best.drop(columns=['_nulls', 'station_rank'])


@flow(name='Get Weather Data (Simple)')
def get_weather_data():
    log = get_run_logger()
    cfg = load_config()

    start = datetime.strptime(cfg['START_DATE'], '%d-%m-%Y')
    end   = datetime.strptime(cfg['END_DATE'],   '%d-%m-%Y')
    tz    = cfg['WEATHER_TIMEZONE']
    url   = get_pg_url()

    stations = london_stations(limit=15)
    if stations.empty:
        log.warning("No stations near London.")
        return
    station_ids = stations['id'].tolist()

    frames = []
    for sid in station_ids:
        try:
            df = fetch_station_hourly(sid, start, end, tz)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log.warning(f"Fetch failed for {sid}: {e}")

    if not frames:
        log.warning("No station returned data.")
        return

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    best = best_per_timestamp(all_rows, station_order=station_ids)

    if best.empty:
        log.warning("No usable rows after selection.")
        return
    
    df.rename(columns={'datetime':'weather_datetime'})
    df_to_pg(best, 'weather_hourly', url, if_exists='replace')
    log.info(f"Loaded {len(best)} rows into weather_hourly")


if __name__ == "__main__":
    get_weather_data()
