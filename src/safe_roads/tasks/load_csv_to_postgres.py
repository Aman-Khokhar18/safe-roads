from pathlib import Path
from typing import Optional, Iterable
import re
import pandas as pd
from prefect import task, get_run_logger
from sqlalchemy import create_engine

from safe_roads.utils.data import df_to_pg


def get_table_name(path: Path) -> str:
    parts = Path(path).stem.split('-')
    for part in parts:
        if part.isnumeric():
            table_year = part
    end = parts[-1].removesuffix('.csv')
    return (table_year + '_' + end)

import pandas as pd

def fix_headers_retry(df: pd.DataFrame) -> pd.DataFrame:
    def count_unnamed(cols) -> int:
        cnt = 0
        for c in cols:
            s = "" if pd.isna(c) else str(c)
            if s.strip() == "" or s.lower().startswith("unnamed"):
                cnt += 1
        return cnt

    tries = 0
    out = df.copy()

    while tries < 4 and len(out) > 0 and count_unnamed(out.columns) > 3:
        # promote first row to header
        new_header = out.iloc[0].astype(str).tolist()
        out = out.iloc[1:].copy()
        out.columns = new_header
        out.reset_index(drop=True, inplace=True)
        tries += 1

    return out


@task(name="Load Collision CSVs into Postgres")
def load_csv_to_pg(csv_path: Path, db_url: str):
    table = get_table_name(csv_path)

    logger = get_run_logger()
    logger.info(f"Loading {csv_path.name} -> {table}")

    df = pd.read_csv(csv_path, encoding="cp1252")
    df = fix_headers_retry(df)
    df["source"] = str(csv_path)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        df.to_sql(
            name=table,
            con=conn,
            schema='public',          
            if_exists="replace",     
            index=False,       
        )

    summary = {
        "file": csv_path.name,
        "table": f"{table}",
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    logger.info(f"Loaded {summary['rows']:,} rows into {summary['table']}")
    return summary

