from sqlalchemy import create_engine, text, inspect
import pandas as pd
from pathlib import Path
import sqlparse
from dotenv import load_dotenv

load_dotenv()

def df_to_pg(df: pd.DataFrame, table: str, db_url: str, if_exists: str | None = "replace", schema: str = "public"):
    engine = create_engine(db_url)
    df.to_sql(name=table, con=engine, schema=schema, if_exists=if_exists, index=False)
    

def load_database(db_url: str, table: str, columns: str | None = None):
    engine = create_engine(db_url)

    if columns:
        columns = ",".join(
            '"' + c.strip().strip('"').strip("'").replace('"', '""') + '"'
            for c in columns.split(",") if c.strip()
        )
        df = pd.read_sql(
            text(f"""SELECT {columns}

                     FROM {table}"""),
            con=engine
        )
    else:
        df = pd.read_sql_table(table, con=engine)
    return df

def _infer_sql_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s): return "BOOLEAN"
    if pd.api.types.is_integer_dtype(s): return "INTEGER"
    if pd.api.types.is_float_dtype(s): return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(s): return "TIMESTAMP"
    return "TEXT"  

def add_df_to_table(df: pd.DataFrame, db_url: str, table: str, if_exists: str):
    columns = df.columns
    engine = create_engine(db_url)

    with engine.begin() as con:
        for column in columns:
            coltype = _infer_sql_type(df[column])
            con.execute(
            text(f"""
                ALTER TABLE "{table}"
                ADD COLUMN IF NOT EXISTS "{column}" {coltype};
            """))

        df.to_sql(table, con, if_exists=if_exists, index=False)


def _run_sql_file(db_url: str, sql_path: Path) -> None:
    engine = create_engine(db_url)

    sql_text = Path(sql_path).read_text(encoding="utf-8")
    statements = [s.strip() for s in sqlparse.split(sql_text) if s.strip()]

    with engine.begin() as conn:  
        for stmt in statements:
            conn.execute(text(stmt))


def _table_exists(url: str, table: str, schema: str = "public") -> bool:
    sql = text("""
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = :schema
            AND table_name   = :table
        )   
                """)
    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        return bool(conn.execute(sql, {"schema": schema, "table": table}).scalar_one())
    
    
def any_attendant_tables(url: str) -> bool:
    sql = text("""
        SELECT EXISTS (
          SELECT 1
          FROM pg_catalog.pg_class c
          JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
          WHERE c.relkind = 'r'
            AND n.nspname NOT IN ('pg_catalog','information_schema')
            AND c.relname ILIKE '%attendant%'
        )
    """)
    eng = create_engine(url)
    with eng.connect() as cn:
        return cn.execute(sql).scalar()