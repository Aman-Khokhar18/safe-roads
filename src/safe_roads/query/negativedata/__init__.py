from importlib.resources import files
import sqlparse
from sqlalchemy import create_engine, text

def read_sql(name: str) -> str:
    return (files(__name__) / name).read_text(encoding="utf-8")

def run_sql(db_url: str, name: str) -> None:
    engine = create_engine(db_url)
    sql_text = read_sql(name)
    stmts = [s.strip() for s in sqlparse.split(sql_text) if s.strip()]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
