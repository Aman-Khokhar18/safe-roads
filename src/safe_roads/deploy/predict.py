import math
import numpy as np
import pandas as pd
import requests

import gc
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, DOUBLE_PRECISION
from prefect import task, get_run_logger

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
    logger = get_run_logger()
    config      = load_config("configs/app.yml")
    table_name  = config['INPUT_TABLE']
    model_api   = config['SAFE_ROADS_API']
    endpoint    = model_api + "/predict"
    batch_size  = int(config['BATCH_SIZE'])
    timeout     = config['HTTP_TIMEOUT']

    # new (optional) knobs; add to config if you like
    read_chunksize  = int(config.get('READ_CHUNKSIZE', 10_000))  # rows pulled from SQL at a time
    write_chunksize = int(config.get('WRITE_CHUNKSIZE', 10_000))  # rows per INSERT batch in to_sql

    OUT_TABLE  = config["OUT_TABLE"]
    IF_EXISTS0 = "replace"

    logger.info(f"Streaming data from table: {table_name} (read_chunksize={read_chunksize})")
    df_iter = data_loader(table_name, chunksize=read_chunksize)

    # In case someone sets chunksize=None by mistake, normalize to an iterable
    if isinstance(df_iter, pd.DataFrame):
        df_iter = [df_iter]

    sess = requests.Session()
    engine = create_engine(url=get_pg_url(), pool_pre_ping=True)

    dtype_map = {
        "h3": VARCHAR(32),
        "parent_h3": VARCHAR(32),
        "prediction": INTEGER(),
        "probability": DOUBLE_PRECISION(),
    }

    first_write = True
    total_rows  = 0
    chunk_idx   = 0

    for chunk_idx, chunk_df in enumerate(df_iter, start=1):
        if chunk_df is None or chunk_df.empty:
            continue

        # Ensure identifier columns exist
        for col in ("h3", "parent_h3"):
            if col not in chunk_df.columns:
                chunk_df[col] = pd.NA

        # Clean infinities/NaNs early
        chunk_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

        logger.info(f"SQL chunk {chunk_idx}: {len(chunk_df)} rows")

        # Batch within the chunk for the model API
        num_batches_in_chunk = math.ceil(len(chunk_df) / batch_size)
        for b in range(num_batches_in_chunk):
            start = b * batch_size
            end   = min(len(chunk_df), (b + 1) * batch_size)

            batch_df = chunk_df.iloc[start:end].copy()
            if batch_df.empty:
                continue

            batch_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

            # JSON payload: scrub to plain Python + None
            batch_records = batch_df.where(batch_df.notna(), None).to_dict(orient="records")
            payload = {"instances": _scrub_jsonable(batch_records)}

            # Call model
            resp = sess.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            out = resp.json()

            preds = out.get("predictions", [])
            probs = out.get("probabilities", [])
            if len(preds) != len(batch_df) or len(probs) != len(batch_df):
                raise RuntimeError("API output length mismatch with input batch")

            out_df = pd.DataFrame({
                "h3": batch_df["h3"].astype("string"),
                "parent_h3": batch_df["parent_h3"].astype("string"),
                "prediction": [int(x) for x in preds],
                "probability": [float(p) for p in probs],
            })

            # Write immediately; keep memory low
            out_df.to_sql(
                OUT_TABLE,
                con=engine,
                if_exists=IF_EXISTS0 if first_write else "append",
                index=False,
                dtype=dtype_map,
                method="multi",
                chunksize=write_chunksize,
            )
            first_write = False

            total_rows += len(out_df)
            logger.info(f"Chunk {chunk_idx} batch {b+1}/{num_batches_in_chunk}: wrote {len(out_df)} rows (total {total_rows})")

            # Free memory
            del out_df, batch_df, batch_records, preds, probs, out
            gc.collect()

    if total_rows == 0:
        raise RuntimeError(f"No rows returned by data_loader('{table_name}')")

    logger.info(f"Done. Wrote {total_rows} rows to {OUT_TABLE}.")
    print(f"Done. Wrote {total_rows} rows to {OUT_TABLE}.")


if __name__ == "__main__":
    predict()
