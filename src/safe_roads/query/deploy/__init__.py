# src/safe_roads/query/deploy/__init__.py
from importlib.resources import files
import sqlparse
from sqlalchemy import create_engine, text

def read_sql(name: str) -> str:
    p = files(__name__) / name
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"SQL resource not found: {p}") from e

def run_sql(db_url: str, name: str) -> None:
    engine = create_engine(db_url)
    sql_text = read_sql(name)

    statements = [s.strip() for s in sqlparse.split(sql_text) if s.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
