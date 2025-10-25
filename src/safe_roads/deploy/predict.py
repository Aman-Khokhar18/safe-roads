import math
import gc
import logging
import math
import numpy as np
import pandas as pd
import requests

from tqdm.auto import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, DOUBLE_PRECISION

from safe_roads.utils.mlutil import data_loader
from safe_roads.utils.config import load_config, get_pg_url


logger = logging.getLogger("safe_roads.predict")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    )


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


def _new_session_with_retries(total=5, backoff=0.5):
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def predict():
    config      = load_config("configs/app.yml")
    table_name  = config["INPUT_TABLE"]
    model_api   = config["SAFE_ROADS_API"]
    endpoint    = model_api.rstrip("/") + "/predict"
    batch_size  = int(config["BATCH_SIZE"])
    timeout     = config["HTTP_TIMEOUT"]

    # optional tunables
    read_chunksize  = int(config.get("READ_CHUNKSIZE", 10_000))
    write_chunksize = int(config.get("WRITE_CHUNKSIZE", 10_000))

    OUT_TABLE  = config["OUT_TABLE"]
    IF_EXISTS0 = "replace"

    # fixed total rows provided by you
    TOTAL_ROWS = 3_159_825

    logger.info("Streaming data from table: %s (read_chunksize=%s)", table_name, read_chunksize)
    df_iter = data_loader(table_name, chunksize=read_chunksize, mode="predict")

    # Normalize to iterable if a full DataFrame is returned
    if isinstance(df_iter, pd.DataFrame):
        df_iter = [df_iter]

    sess = _new_session_with_retries()
    engine = create_engine(url=get_pg_url(), pool_pre_ping=True)

    dtype_map = {
        "h3": VARCHAR(32),
        "parent_h3": VARCHAR(32),
        "prediction": INTEGER(),
        "probability": DOUBLE_PRECISION(),
    }

    first_write = True
    total_rows_written = 0

    # single global progress bar in ROWS
    pbar = tqdm(total=TOTAL_ROWS, desc="Predicting", unit="rows")

    for chunk_idx, chunk_df in enumerate(df_iter, start=1):
        if chunk_df is None or chunk_df.empty:
            continue

        # Ensure identifier columns exist
        for col in ("h3", "parent_h3"):
            if col not in chunk_df.columns:
                chunk_df[col] = pd.NA

        # Clean infinities/NaNs early
        chunk_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

        rows_in_chunk = len(chunk_df)
        num_batches_in_chunk = math.ceil(rows_in_chunk / batch_size)

        for b in range(num_batches_in_chunk):
            start = b * batch_size
            end   = min(rows_in_chunk, (b + 1) * batch_size)

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

            # Validate response & extract
            if "probabilities" not in out:
                raise RuntimeError(f"Bad API response (missing 'probabilities'): {out}")

            probs = out["probabilities"]
            # Normalize if server returns list-of-dicts
            if probs and isinstance(probs[0], dict):
                key = "probability" if "probability" in probs[0] else "1" if "1" in probs[0] else None
                if key is None:
                    raise RuntimeError(f"Unrecognized probability schema: {probs[0]}")
                probs = [float(p[key]) for p in probs]

            if len(probs) != len(batch_df):
                raise RuntimeError(
                    f"API output length mismatch: probs={len(probs)} vs batch={len(batch_df)}"
                )

            out_df = pd.DataFrame({
                "h3": batch_df["h3"].astype("string"),
                "parent_h3": batch_df["parent_h3"].astype("string"),
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

            wrote = len(out_df)
            total_rows_written += wrote
            pbar.update(wrote)  # advance by rows processed in this batch

            # Free memory
            del out_df, batch_df, batch_records, probs, out
            gc.collect()

    pbar.close()

    if total_rows_written == 0:
        raise RuntimeError(f"No rows returned by data_loader('{table_name}')")

    logger.info("Done. Wrote %d rows to %s.", total_rows_written, OUT_TABLE)
    print(f"Done. Wrote {total_rows_written} rows to {OUT_TABLE}.")


if __name__ == "__main__":
    predict()
