from importlib.resources import files
import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import (
    OperationalError,
    DBAPIError,
    NoSuchModuleError,
    ArgumentError,
    InterfaceError,
)
from sqlalchemy.engine.url import make_url


def read_sql(name: str) -> str:
    p = files(__name__) / name
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"SQL resource not found: {p}") from e


def _safe_url_str(db_url: str) -> str:
    try:
        u = make_url(db_url)
        return str(u.set(password="***"))
    except Exception:
        return "<redacted>"


def run_sql(db_url: str, name: str, *, connect_timeout: int = 5) -> None:
    connect_args = {}
    try:
        backend = make_url(db_url).get_backend_name()
        if backend in ("postgresql", "mysql", "mariadb"):
            connect_args["connect_timeout"] = connect_timeout
    except Exception:
        pass  

    try:
        engine = create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)
    except (NoSuchModuleError, ArgumentError) as e:
        raise ValueError(f"Invalid DB URL or missing driver: {e}") from e

    sql_text = read_sql(name)
    statements = [s.strip() for s in sqlparse.split(sql_text) if s.strip()]

    safe_url = _safe_url_str(db_url)

    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1")) 
            for stmt in statements:
                conn.execute(text(stmt))
    except (OperationalError, InterfaceError) as e:
        raise ConnectionError(
            f"Database at {safe_url} is unreachable (network/host/auth). "
            f"Original error: {e.__class__.__name__}: {e}"
        ) from e
    except DBAPIError as e:
        if getattr(e, "connection_invalidated", False):
            raise ConnectionError(
                f"Lost/invalid connection to {safe_url}: {e.__class__.__name__}: {e}"
            ) from e
        raise
