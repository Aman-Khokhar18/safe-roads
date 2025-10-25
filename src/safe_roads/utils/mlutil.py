
import pandas as pd
from sqlalchemy import text, create_engine
from typing import Iterable, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
from safe_roads.utils.config import get_pg_url, load_config
from sklearn.model_selection import train_test_split


def data_loader(table, chunksize: int | None = None, mode: str | None = 'train'):

    mode = mode.lower()

    if mode not in {"train", "predict"}:
        raise ValueError("mode must be 'train' or 'predict'")

    url = get_pg_url()
    engine = create_engine(url, pool_pre_ping=True)
    order_by = "h3, year, month, day, hour"

    collision = "cd.collision," if mode == "train" else ""
    query = f"""
        SELECT
            cd.h3, cd.parent_h3, cd.highway, cd.lanes, cd.width, cd.surface, cd.smoothness, cd.oneway, cd.junction,
            cd.traffic_signals, cd.traffic_calming, cd.crossing, cd.sidewalk, cd.cycleway, cd.bicycle, cd.lit,
            cd.access, cd.vehicle, cd.hgv, cd.psv, cd.bus, cd.overtaking, cd.bridge, cd.tunnel, cd.layer, cd.incline,
            cd.barrier, cd.amenity, cd.year, cd.month, cd.day, cd.hour, cd.is_weekend, cd.road_class3, cd.borough,
            cd.maxspeed_mph, cd.is_junction, cd.is_turn, cd.junction_degree, cd.temp, cd.dwpt, cd.rhum, cd.prcp,
            cd.snow, cd.wdir, cd.wspd, cd.wpgt, cd.pres, cd.tsun, cd.coco, cd.hour_sin, cd.hour_cos, cd.dow_sin, cd.dow_cos,
            cd.dom_sin, cd.dom_cos, cd.month_sin, cd.month_cos, cd.traffic_light_count, cd.crossing_count,
            cd.motorway_other_count, cd.cycleway_count, {collision}

            COALESCE(lyw.h3_collisions_last_year, 0)  AS h3_collisions_last_year,
            COALESCE(lyw.n1_collisions_last_year, 0)  AS n1_collisions_last_year,
            COALESCE(lyw.n2_collisions_last_year, 0)  AS n2_collisions_last_year,
            COALESCE(lyw.n3_collisions_last_year, 0)  AS n3_collisions_last_year,
            COALESCE(lyw.n4_collisions_last_year, 0)  AS n4_collisions_last_year,
            COALESCE(lyw.n5_collisions_last_year, 0)  AS n5_collisions_last_year,
            COALESCE(lyw.n6_collisions_last_year, 0)  AS n6_collisions_last_year,
            COALESCE(lyw.parent_collisions_last_year, 0) AS parent_collisions_last_year
        FROM {table} AS cd
        LEFT JOIN public.h3_last_year_collisions_wide AS lyw
        ON lyw.h3 = cd.h3
        AND lyw.year = cd.year
        ORDER BY {order_by}
    """


    with engine.connect().execution_options(stream_results=True) as conn:
        it = pd.read_sql_query(text(query), conn, chunksize=chunksize)
        if isinstance(it, pd.DataFrame):
            yield it
        else:
            for chunk in it:
                yield chunk
                

def ensure_types(config, data):
    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]

    for col in CATEGORICAL:
        if col in data.columns: 
            data[col] = data[col].astype("string").fillna("__MISSING__")
    for col in BOOLEAN:
        if col in data.columns and pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype("int8")

    for col in NUMERICAL:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")



def prepare_data(config: dict, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    RANDOM_STATE = config["RANDOM_STATE"]
    TEST_SIZE    = config["TEST_SIZE"]
    VAL_SIZE     = config["VAL_SIZE"]
    TARGET       = config["TARGET"]

    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]
    features    = NUMERICAL + CATEGORICAL + BOOLEAN

    df = data.copy()
    ensure_types(config, df)

    X = df.loc[:, features].copy()
    y = df.loc[:, TARGET].copy()

    # Use stratification if classification labels are available (>=2 classes)
    stratify_main = y if y.nunique() > 1 else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_main
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    stratify_val = y_temp if (stratify_main is not None and y_temp.nunique() > 1) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=stratify_val
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



