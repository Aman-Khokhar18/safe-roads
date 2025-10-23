
import pandas as pd
from sqlalchemy import text, create_engine
from typing import Iterable, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
from safe_roads.utils.config import get_pg_url, load_config
from sklearn.model_selection import train_test_split


def data_loader(table, chunksize: int | None = None):
    url = get_pg_url()
    engine = create_engine(url, pool_pre_ping=True)

    order_by = "h3, year, month, day, hour"
    query = f"""
        SELECT
            h3, neighbor_1, neighbor_2, neighbor_3, neighbor_4, neighbor_5, neighbor_6,
            parent_h3, name, highway, lanes, width, surface, smoothness, oneway, junction,
            traffic_signals, traffic_calming, crossing, sidewalk, cycleway, bicycle, lit,
            access, vehicle, hgv, psv, bus, overtaking, bridge, tunnel, layer, incline,
            barrier, amenity, year, month, day, hour, is_weekend, road_class3, borough,
            maxspeed_mph, is_junction, is_turn, junction_degree, temp, dwpt, rhum, prcp,
            snow, wdir, wspd, wpgt, pres, tsun, coco, hour_sin, hour_cos, dow_sin, dow_cos,
            dom_sin, dom_cos, month_sin, month_cos, traffic_light_count, crossing_count,
            motorway_other_count, cycleway_count
        FROM {table}
        ORDER BY {order_by}
    """

    # Keep the connection + transaction open while consuming the iterator
    with engine.connect().execution_options(stream_results=True) as conn:
        it = pd.read_sql_query(text(query), conn, chunksize=chunksize)
        # Normalize to an iterator even if chunksize is None
        if isinstance(it, pd.DataFrame):
            yield it
        else:
            for chunk in it:
                yield chunk
                

def handle_nans(config, data):
    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]

    for col in CATEGORICAL:
        if col in data.columns:
            data[col] = data[col].astype("string").fillna("<NA>")

    for col in BOOLEAN:
        if col in data.columns and pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype("int8")

    for col in NUMERICAL:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")



def prepare_data(config: dict, pos: pd.DataFrame, neg: pd.DataFrame):
    """
    Recreate the exact splits used during training based on config.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    RANDOM_STATE = config["RANDOM_STATE"]
    TEST_SIZE    = config["TEST_SIZE"]
    VAL_SIZE     = config["VAL_SIZE"]
    TARGET       = config["TARGET"]

    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]
    features    = NUMERICAL + CATEGORICAL + BOOLEAN

    pos = pos.copy()
    neg = neg.copy()
    pos[TARGET] = config["POS_LABEL"]
    neg[TARGET] = config["NEG_LABEL"]

    data = pd.concat([pos, neg], axis=0, ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    handle_nans(config, data)

    X = data.loc[:, features].copy()
    y = data.loc[:, TARGET].copy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



