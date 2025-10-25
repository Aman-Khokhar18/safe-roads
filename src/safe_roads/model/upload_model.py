from huggingface_hub import HfApi, create_repo, create_commit, CommitOperationAdd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = "AmanKhokhar/safe-roads"
LOCAL_MODEL_DIR = Path("artifacts/catboost_model")
TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
RUN_ID = "91122868ad3e4c4f8314cc6db54b7d6e"
COMMIT_MSG = f"Upload selected CatBoost files from MLflow run: {RUN_ID}"

api = HfApi(token=TOKEN)
print("whoami:", api.whoami())

create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)

src_model    = LOCAL_MODEL_DIR / f"model_{RUN_ID}.cbm"
src_features = LOCAL_MODEL_DIR / f"features_{RUN_ID}.json"
src_meta     = LOCAL_MODEL_DIR / f"model_meta_{RUN_ID}.json"

# Sanity checks
for p in [src_model, src_features, src_meta]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

# Upload with new names/paths in the repo
ops = [
    CommitOperationAdd(
        path_in_repo="model.cbm",          
        path_or_fileobj=str(src_model)
    ),
    CommitOperationAdd(
        path_in_repo="features.json",             
        path_or_fileobj=str(src_features)
    ),
    CommitOperationAdd(
        path_in_repo="model_meta.json",
        path_or_fileobj=str(src_meta)
    ),
]

commit = create_commit(
    repo_id=HF_REPO_ID,
    repo_type="model",
    operations=ops,
    commit_message=COMMIT_MSG,
    token=TOKEN,
)

print(f"Done. Commit: {commit.oid}")
print(f"Browse: https://huggingface.co/{HF_REPO_ID}/tree/{commit.oid}")
