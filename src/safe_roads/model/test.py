from pathlib import Path
import glob
import json
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb

from safe_roads.utils.config import load_config
from safe_roads.utils.mlutil import get_pg_url

# ------------------ SETTINGS ------------------
TABLE          = "osm_deploy_latest_w"                 # live source table
OUT_PRED_TABLE = f"{TABLE}_predictions"                # new table to write
PRED_COL       = "xgb_score"                           # probability column name
ARTIFACTS_DIR  = "artifacts/xgboost_model"             # where model_*.json etc. are
RUN_ID         = "50d1896a75d94a4caa8184f4aa273cd3"    # set None for latest valid
# ------------------------------------------------

def _pick_latest(path_glob: str) -> Path:
    cand = [Path(p) for p in glob.glob(path_glob)]
    if not cand:
        raise FileNotFoundError(f"No files for pattern: {path_glob}")
    return max(cand, key=lambda p: p.stat().st_mtime)

def _pick_valid_model(artifacts_dir: str, run_id: str | None) -> Path:
    """Pick a valid model_*.json (exclude model_meta_*.json), newest-first."""
    ad = Path(artifacts_dir)
    patterns = []
    if run_id:
        patterns.append(str(ad / f"model_*{run_id}*.json"))
    patterns.append(str(ad / "model_*.json"))
    for pat in patterns:
        for s in sorted(glob.glob(pat), key=lambda x: Path(x).stat().st_mtime, reverse=True):
            p = Path(s)
            if p.name.startswith("model_meta_"):
                continue
            txt = p.read_text(encoding="utf-8").strip()
            if len(txt) < 200 or txt.lower() == "null":
                continue
            try:
                json.loads(txt)  # sanity parse
                return p
            except Exception:
                continue
    raise FileNotFoundError("No valid model_*.json found.")

def _load_artifacts(artifacts_dir: str, run_id: str | None):
    ad = Path(artifacts_dir)
    model_path = _pick_valid_model(artifacts_dir, run_id)
    feats_path = _pick_latest(str(ad / "features_*.json"))
    meta_path  = _pick_latest(str(ad / "model_meta_*.json"))
    features   = json.loads(Path(feats_path).read_text(encoding="utf-8"))
    model_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    print(f"[using model] {model_path}")
    print(f"[using feats ] {feats_path}")
    print(f"[using meta  ] {meta_path}")
    return Path(model_path), features, model_meta

def score_and_save():
    # 1) config + artifacts
    config = load_config("configs/train.yml")
    categorical = list(config.get("CATEGORICAL", []))
    numerical   = list(config.get("NUMERICAL",   []))
    boolean     = list(config.get("BOOLEAN",     []))

    model_path, features, model_meta = _load_artifacts(ARTIFACTS_DIR, RUN_ID)

    # 2) load XGBoost Booster (3.0.5)
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # 3) read ALL columns from live table
    engine = create_engine(get_pg_url())
    df_src = pd.read_sql(f'SELECT * FROM {TABLE}', con=engine)

    # 4) build X from training-time features (create missing as 0; keep exact order)
    X = df_src.copy()
    for c in features:
        if c not in X.columns:
            X[c] = 0
    X = X[features]

    # 5) DTYPE FIXES — enforce train-time dtypes
    # categoricals
    for c in categorical:
        if c in X.columns:
            if X[c].dtype == "object":
                X[c] = X[c].astype("string").astype("category")
            elif not pd.api.types.is_categorical_dtype(X[c]):
                X[c] = X[c].astype("category")
    # booleans
    for c in boolean:
        if c in X.columns:
            # normalize common truthy/falsey strings and numbers
            col = X[c]
            if col.dtype == "object":
                col = col.astype("string").str.lower()
                col = col.map({
                    "true": True, "1": True, "t": True, "y": True,
                    "false": False, "0": False, "f": False, "n": False
                })
            X[c] = col.astype("boolean")
    # numericals
    for c in numerical:
        if c in X.columns and X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # any remaining object columns that aren't declared categorical -> try numeric
    for c in [cn for cn in X.columns if X[cn].dtype == "object"]:
        if c in categorical:
            X[c] = X[c].astype("string").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # 6) predict probabilities — use all trees in saved booster (no iteration_range)
    dtest = xgb.DMatrix(X, enable_categorical=True)
    probs = booster.predict(dtest)

    # 7) write full copy with prediction column appended
    df_out = df_src.copy()
    df_out[PRED_COL] = probs
    df_out.to_sql(OUT_PRED_TABLE, con=engine, if_exists="replace", index=False)
    print(f"Wrote predictions to {OUT_PRED_TABLE} with column {PRED_COL}")

if __name__ == "__main__":
    score_and_save()
