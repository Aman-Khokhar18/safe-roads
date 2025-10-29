import os, json, gzip, io, base64, requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
DB_DSN = os.getenv("DATABASE_URL")
PREDICTION_TABLE = "prediction"

TOKEN = os.environ["GITHUB_TOKEN"]  
REPO_FULL = "Aman-Khokhar18/safe-roads-london"

BRANCH = "main"
PATH_IN_REPO = "assets/backend/predictions.json.gz"
API = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

def commit_to_git():
    engine = create_engine(DB_DSN, pool_pre_ping=True, future=True)
    sql_preds = text(f"""
        SELECT
            h3::text            AS h3,
            probability::float  AS probability
        FROM {PREDICTION_TABLE}
        WHERE probability IS NOT NULL
    """)

    with engine.begin() as conn:
        df = pd.read_sql(sql_preds, conn)

    if df.empty:
        raise SystemExit("No rows returned. Check table/columns/SQL.")

    df = df.dropna(subset=["h3", "probability"]).copy()
    df["probability"] = df["probability"].clip(lower=0.0, upper=1.0)

    records = list(map(list, zip(
        df["h3"].astype(str).tolist(),
        df["probability"].astype(float).tolist()
    )))

    # --- Get latest weather_datetime ---
    sql_latest_weather = text("""
        SELECT retrieved_at_utc
        FROM weather_live
        ORDER BY retrieved_at_utc DESC
        LIMIT 1
    """)

    with engine.begin() as conn:
        val = conn.execute(sql_latest_weather).scalar_one_or_none()

    if val is None:
        weather_text = None
    elif isinstance(val, datetime):
        weather_text = val.isoformat()
    else:
        weather_text = str(val)

    # --- Payload ---
    payload = {
        "data": records,                   
        "meta": {"weather_datetime": weather_text}
    }

    # --- Serialize + gzip to memory ---
    raw_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw_json)
    content_bytes = buf.getvalue()
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    # --- Get current file SHA if it exists (required for updates) ---
    contents_url = f"{API}/repos/{REPO_FULL}/contents/{PATH_IN_REPO}"
    params = {"ref": BRANCH}
    r = requests.get(contents_url, headers=HEADERS, params=params)
    if r.status_code == 200:
        sha = r.json().get("sha")
    elif r.status_code == 404:
        sha = None
    else:
        raise RuntimeError(f"Failed to check existing file: {r.status_code} {r.text}")

    # --- Commit to GitHub ---
    commit_msg = (
        f"chore(data): update predictions.json.gz "
        f"({len(records)} rows; weather_datetime={weather_text})"
    )

    put_payload = {
        "message": commit_msg,
        "content": content_b64,
        "branch": BRANCH,
        # Optional: set committer identity; otherwise defaults to token/app identity
        # "committer": {"name": "Data Bot", "email": "actions@github.com"},
    }
    if sha:
        put_payload["sha"] = sha

    resp = requests.put(contents_url, headers=HEADERS, json=put_payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Helpful hint if branch protection or permissions block the push
        raise RuntimeError(f"GitHub commit failed: {resp.status_code} {resp.text}") from e

    out = resp.json()
    commit_url = out.get("commit", {}).get("html_url")
    print(f"Committed {PATH_IN_REPO} to {REPO_FULL}@{BRANCH} "
        f"rows={len(records)} meta.weather_datetime={weather_text}\n{commit_url}")

if __name__ == "__main__":
    commit_to_git()