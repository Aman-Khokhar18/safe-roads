import math
import numpy as np
import pandas as pd
import requests

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, DOUBLE_PRECISION
from prefect import task

from safe_roads.utils.mlutil import data_loader
from safe_roads.utils.config import load_config, get_pg_url



def _scrub_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _scrub_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_jsonable(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@task
def predict():
    config = load_config("configs/app.yml")
    table_name = config['INPUT_TABLE']
    model_api = config['SAFE_ROADS_API']
    endpoint   = model_api + "/predict"
    batch_size = config['BATCH_SIZE']
    timeout    = config['HTTP_TIMEOUT'] 

    OUT_TABLE  = config["OUT_TABLE"]
    IF_EXISTS0 = "replace"

    df = data_loader(table_name)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No rows returned by data_loader('{table_name}')")
    n = len(df)

    # Keep identifiers
    for col in ("h3", "parent_h3"):
        if col not in df.columns:
            df[col] = pd.NA

    df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    sess = requests.Session()

    url = "postgresql+psycopg://amank:london_123@saferoads.cg5mscywwjft.us-east-1.rds.amazonaws.com:5432/saferoads?sslmode=require"

    engine = create_engine(url=url, pool_pre_ping=True)
    dtype_map = {
        "h3": VARCHAR(32),
        "parent_h3": VARCHAR(32),
        "prediction": INTEGER(),
        "probability": DOUBLE_PRECISION(),
    }

    # Batching
    num_batches = math.ceil(n / batch_size)
    print(f"Scoring {n} rows via {endpoint} in {num_batches} batches (size {batch_size}).")

    first = True
    for b in range(num_batches):
        start = b * batch_size
        end   = min(n, (b + 1) * batch_size)

        batch_df = df.iloc[start:end].copy()
        batch_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

        # JSON payload
        batch_records = batch_df.where(batch_df.notna(), None).to_dict(orient="records")
        payload = {"instances": _scrub_jsonable(batch_records)}

        # Call API
        resp = sess.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        out = resp.json()

        preds = out.get("predictions", [])
        probs = out.get("probabilities", [])

        if len(preds) != (end - start) or len(probs) != (end - start):
            raise RuntimeError("API output length mismatch with input batch")

        out_df = pd.DataFrame({
            "h3": df.iloc[start:end]["h3"].astype("string"),
            "parent_h3": df.iloc[start:end]["parent_h3"].astype("string"),
            "prediction": [int(x) for x in preds],
            "probability": [float(p) for p in probs],
        })

        # Write to Postgres
        out_df.to_sql(
            OUT_TABLE,
            con=engine,
            if_exists=IF_EXISTS0 if first else "append",
            index=False,
            dtype=dtype_map,
            method="multi",
        )
        first = False

        print(f"Batch {b+1}/{num_batches}: wrote {len(out_df)} rows")

    print(f"Done. Wrote {n} rows to {OUT_TABLE}.")


if __name__ == "__main__":
    predict()
