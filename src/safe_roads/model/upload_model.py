
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = "AmanKhokhar/safe-roads"  
LOCAL_MODEL_DIR = Path("artifacts/catboost_model")       
COMMIT_MSG = "Upload existing CatBoost artifacts (no retrain)"
TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")              

api = HfApi(token=TOKEN)
print("whoami:", api.whoami())  

create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)
# Upload the folder
commit = upload_folder(
    repo_id=HF_REPO_ID,
    repo_type="model",
    folder_path=str(LOCAL_MODEL_DIR),
    commit_message=COMMIT_MSG,
    token=TOKEN,
)

print(f"Done. Commit: {commit.oid}")
print(f"Browse: https://huggingface.co/{HF_REPO_ID}/tree/{commit.oid}")
