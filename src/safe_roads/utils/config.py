import yaml
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
load_dotenv()

DEFAULT_PATHS = (Path("configs/config.yaml"), Path("configs/config.yml"))


# Checks if default path exists in list of default paths
def _pick_default_path():
    for p in DEFAULT_PATHS:
        if p.exists():
            return p
    return DEFAULT_PATHS[0]

# Loads .yaml file from given file path else chooses default path
def load_config(path: str | Path | None = None):

    p = Path(path) if path else _pick_default_path()

    try:
        with p.open("r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found at {path}") from e
    
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {path}: {e}") from e
    
    return config

def get_pg_url():
    user = os.getenv("POSTGRES_USER")
    pwd  = os.getenv("POSTGRES_PASSWORD")
    db   = os.getenv("POSTGRES_DB")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    sslmode = "require"
    return f"postgresql://{user}:{quote_plus(pwd)}@{host}:{port}/{db}?sslmode={sslmode}"


def year_month(date_str: str) -> tuple[int, int]:
    dt = datetime.strptime(date_str.strip(), "%d-%m-%Y")
    return dt.year, dt.month

if __name__ == "__main__":
    print(get_pg_url())