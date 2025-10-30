import os, json, gzip, io, base64, requests, tempfile, logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger("safe_roads.transform")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    )

load_dotenv()
DB_DSN = os.getenv("DATABASE_URL")
PREDICTION_TABLE = "prediction"

TOKEN = os.environ["G_TOKEN"]
REPO_FULL = "Aman-Khokhar18/safe-roads-london"

BRANCH = "main"
PATH_IN_REPO = "assets/backend/predictions.json.gz"
API = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

SQL_PREDS = text(f"""
    SELECT
        h3::text            AS h3,
        probability::float  AS probability
    FROM {PREDICTION_TABLE}
    WHERE probability IS NOT NULL
""")

SQL_LATEST_WEATHER = text("""
    SELECT retrieved_at_utc
    FROM weather_live
    ORDER BY retrieved_at_utc DESC
    LIMIT 1
""")

def _get_weather_text(conn):
    val = conn.execute(SQL_LATEST_WEATHER).scalar_one_or_none()
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)

def _get_existing_sha():
    contents_url = f"{API}/repos/{REPO_FULL}/contents/{PATH_IN_REPO}"
    r = requests.get(contents_url, headers=HEADERS, params={"ref": BRANCH})
    if r.status_code == 200:
        return r.json().get("sha")
    if r.status_code == 404:
        return None
    raise RuntimeError(f"Failed to check existing file: {r.status_code} {r.text}")

def _put_file(content_b64, sha, commit_msg):
    contents_url = f"{API}/repos/{REPO_FULL}/contents/{PATH_IN_REPO}"
    payload = {"message": commit_msg, "content": content_b64, "branch": BRANCH}
    if sha:
        payload["sha"] = sha
    resp = requests.put(contents_url, headers=HEADERS, json=payload)
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"message": resp.text}
        raise RuntimeError(
            f"GitHub commit failed: {resp.status_code} "
            f"message={err.get('message')} doc={err.get('documentation_url')}"
        )
    return resp.json()

def commit_to_git():
    engine = create_engine(DB_DSN, pool_pre_ping=True, future=True)
    tmp_path = None
    rows = 0

    with engine.begin() as conn:
        weather_text = _get_weather_text(conn)
        logger.info(f"Streaming rows from {PREDICTION_TABLE}")

        # Stream DB → gzip JSON (to temp file to keep RAM low)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json.gz") as tmpf:
            tmp_path = tmpf.name
            with gzip.GzipFile(fileobj=tmpf, mode="wb") as gz:
                gz.write(b'{"data":[')
                first = True

                # server-side streaming
                result = conn.execution_options(stream_results=True).execute(SQL_PREDS)
                for row in result:
                    h3 = row.h3
                    p = row.probability
                    if h3 is None or p is None:
                        continue
                    # clamp without pandas
                    if p < 0.0: p = 0.0
                    elif p > 1.0: p = 1.0

                    item = json.dumps([str(h3), float(p)], separators=(",", ":")).encode("utf-8")
                    if not first:
                        gz.write(b",")
                    gz.write(item)
                    first = False
                    rows += 1

                meta = {"weather_datetime": weather_text}
                gz.write(b'], "meta":' + json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"}")

    if rows == 0:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SystemExit("No rows returned. Check table/columns/SQL.")

    # Read compressed bytes and upload
    try:
        with open(tmp_path, "rb") as f:
            content_bytes = f.read()
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    sha = _get_existing_sha()
    commit_msg = f"chore(data): update predictions.json.gz ({rows} rows; weather_datetime={weather_text})"
    out = _put_file(content_b64, sha, commit_msg)

    commit_url = out.get("commit", {}).get("html_url")
    logger.info(
        f"Committed {PATH_IN_REPO} to {REPO_FULL}@{BRANCH} "
        f"rows={rows} meta.weather_datetime={weather_text}\n{commit_url}"
    )

if __name__ == "__main__":
    commit_to_git()
