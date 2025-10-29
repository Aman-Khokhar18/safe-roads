from pathlib import Path
from urllib.parse import urljoin
from itertools import product
import requests
from prefect import task, get_run_logger

from safe_roads.utils.config import load_config
from safe_roads.utils.io import download_file


# Typical TFL Data format: <year><fill><type><ext>
def build_filename(year: str, fill: str, type: str, ext: str) -> str:
    return f"{year}{fill}{type}{ext}"


def build_url(base: str, filename: str) -> str:
    return urljoin(base + "/", filename)


@task(name="Check URL")
def check_url(url: str) -> bool:
    log = get_run_logger()
    log.info(f"Checking URL: {url}")

    config = load_config()
    timeout = config['TIMEOUT']
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if 200 <= r.status_code < 400:
            return True
        
        if r.status_code in (403, 405):
            g = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
            try:
                next(g.iter_content(chunk_size=1024))
                return 200 <= g.status_code < 400
            finally:
                g.close()

        log.warning(f"HEAD returned {r.status_code} for {url}")
        return False
    
    except requests.RequestException as e:
        log.warning(f"URL check failed: {e}")
        return False



@task(name="Download file")
def download(url: str, outdir: Path) -> Path:
    return download_file(url=url, outdir=outdir)


# Downloads TFL Date , required fields <year> <type>
# Type : attendent, casuality, vehicle 
def download_collision_file(
    year: str | int,
    file_type: str,
    outdir: str | Path,
) -> Path:
    log = get_run_logger()
    config = load_config()

    base = config["BASE_COLLISION_DATA_URL"].strip()
    ext  = config["COLLISION_DATA_EXTENSION"]
    fills = [
        config["COLLISION_DATA_URL_FILL"].strip(),  # your primary fill
        "-data-",
        "-data-files-",
    ]

    year_str = str(year).strip()
    year_tokens = [year_str, f"jan-dec-{year_str}"]  

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    last_url = None
    for ytok, fill in product(year_tokens, fills):
        filename = build_filename(ytok, fill, file_type, ext)
        url = build_url(base, filename)
        last_url = url

        log.info(f"Trying filename: {filename}")
        if check_url(url):
            log.info(f"URL OK: {url}")
            saved_path = download(url, outdir)
            log.info(f"Saved to: {saved_path.resolve()}")
            return saved_path

    # If we get here, all combos failed
    raise RuntimeError(f"No reachable URL found. Last tried: {last_url}")