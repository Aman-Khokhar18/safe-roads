
import pandas as pd
from sqlalchemy import text, create_engine
from typing import Iterable, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
from safe_roads.utils.config import get_pg_url, load_config
from sklearn.model_selection import train_test_split

config = load_config("configs/train.yml")

def data_loader(table, chunksize: int | None = None, mode: str | None = None):

    url = get_pg_url()
    engine = create_engine(url, pool_pre_ping=True)
    order_by = "h3, year, dt_month, dt_day, dt_hour"

    collision_target = ",cd.collision AS collision_target" if mode == "train" else ""

    query = f"""
        SELECT
            cd.*

            {collision_target}
        FROM {table} AS cd
        ORDER BY {order_by}
    """
    with engine.connect().execution_options(stream_results=True) as conn:
        if chunksize is None:
            df = pd.read_sql_query(text(query), conn)  
            yield df
        else:
            for chunk in pd.read_sql_query(text(query), conn, chunksize=chunksize):
                yield chunk
                

def ensure_types(config, data, model_type: str):

    if model_type not in ["xgboost", "catboost"]:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]

    if model_type == "xgboost":
        for col in CATEGORICAL:
            if col in data.columns: 
                data[col] = data[col].astype("category")

    elif model_type == "catboost":
        for col in CATEGORICAL:
            if col in data.columns:
                data[col] = data[col].astype("string").fillna("__MISSING__")
                
    for col in BOOLEAN:
        if col in data.columns and pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype("int8")

    for col in NUMERICAL:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")


def prepare_data(config: dict, data: pd.DataFrame):
    TARGET      = config["TARGET"]
    NUMERICAL   = config["NUMERICAL"]
    CATEGORICAL = config["CATEGORICAL"]
    BOOLEAN     = config["BOOLEAN"]
    MODEL       = config["MODEL"]
    features    = NUMERICAL + CATEGORICAL + BOOLEAN

    if "dt_year" not in data.columns:
        raise ValueError("Need 'year' column for year-based split.")

    ensure_types(config, data, MODEL)

    spl = config["SPLIT"]
    train_before = int(spl["TRAIN_BEFORE_YEAR"])
    test_year    = int(spl["TEST_YEAR"])
    val_year     = int(spl.get("VAL_YEAR", train_before - 1))

    train_mask = (data["dt_year"] < train_before) & (data["dt_year"] != val_year)
    val_mask   = (data["dt_year"] == val_year)
    test_mask  = (data["dt_year"] == test_year)

    train_df = data.loc[train_mask].copy()
    val_df   = data.loc[val_mask].copy()
    test_df  = data.loc[test_mask].copy()

    X_train, y_train = train_df[features], train_df[TARGET]
    X_val,   y_val   = val_df[features],   val_df[TARGET]
    X_test,  y_test  = test_df[features],  test_df[TARGET]
    return X_train, y_train, X_val, y_val, X_test, y_test




