import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, average_precision_score, roc_auc_score, log_loss
)
from prefect import flow

import mlflow
from xgboost import XGBClassifier

from safe_roads.utils.mlutil import data_loader, prepare_data
from safe_roads.utils.config import load_config


@flow(name="Train XGBoost model")
def train():
    config = load_config("configs/train.yml")
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    RANDOM_STATE = config["RANDOM_STATE"]
    EARLY_STOPPING_ROUNDS = config["EARLY_STOPPING_ROUNDS"]

    # === Artifacts directory (XGBoost) ===
    LOCAL_MODEL_DIR = Path(config.get("LOCAL_MODEL_DIR", "artifacts/xgboost_model"))

    mlflow.set_tracking_uri("http://localhost:5000")

    # --- Data ---
    data = next(data_loader("combined_dataset", chunksize=None, mode="train"))
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, data)

    cat_cols = [c for c in config["CATEGORICAL"] if c in X_train.columns]
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X_train[c]):
            X_train[c] = X_train[c].astype("category")
            X_val[c]   = X_val[c].astype("category")
            X_test[c]  = X_test[c].astype("category")

    # --- Class weight handling ---
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    # --- Model ---
    model = XGBClassifier(
        tree_method="hist",
        device="cuda",                   
        objective="binary:logistic",       
        eval_metric="logloss",        
        n_estimators=15000,          
        learning_rate=0.02,           
        max_depth=6,                  
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        enable_categorical=True,    
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    )

    # --- MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost_binary") as run:
        run_id = run.info.run_id

        # Log small/metadata to MLflow
        mlflow.log_text(json.dumps(list(X_train.columns), indent=2), "features.json")
        mlflow.log_dict(model.get_params(), f"xgboost_params_{run_id}.json")
        mlflow.log_param("scale_pos_weight_auto", scale_pos_weight)

        # --- Train with early stopping ---
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # Best iteration 
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None:
            best_iter = getattr(model, "best_ntree_limit", None)
        if best_iter is None:
            best_iter = model.n_estimators

        # --- Probability metrics ---
        test_probs = model.predict_proba(X_test)[:, 1]

        brier = brier_score_loss(y_test, test_probs)
        auc = roc_auc_score(y_test, test_probs)
        ap = average_precision_score(y_test, test_probs)   # PRAUC
        logloss = log_loss(y_test, test_probs, labels=[0, 1])

        pi = float(np.mean(y_test))
        bss = 1.0 - brier / (pi * (1.0 - pi)) if 0.0 < pi < 1.0 else np.nan

        mlflow.log_metrics({
            "AUC": auc,
            "PRAUC": ap,
            "LogLoss": logloss,
            "brier": brier,
            "bss": bss
        })

        # ---------- Local save ----------
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = LOCAL_MODEL_DIR / f"model_{run_id}.json"   # JSON is portable/new format
        model.save_model(model_path)

        # Save features list and meta similar to your CatBoost flow
        (LOCAL_MODEL_DIR / f"features_{run_id}.json").write_text(
            json.dumps(list(X_train.columns), indent=2)
        )

        # For compatibility with your downstream loaders
        name_to_idx = {n: i for i, n in enumerate(list(X_train.columns))}
        cat_feature_indices = [name_to_idx[c] for c in cat_cols if c in name_to_idx]

        meta = {
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "CATEGORICAL": config["CATEGORICAL"],
            "NUMERICAL":  config["NUMERICAL"],
            "BOOLEAN":    config["BOOLEAN"],
            "cat_feature_indices": cat_feature_indices,
            "enable_categorical": True,
        }
        (LOCAL_MODEL_DIR / f"model_meta_{run_id}.json").write_text(json.dumps(meta, indent=2))

        # Log directory as artifacts
        mlflow.log_artifacts(str(LOCAL_MODEL_DIR), artifact_path=f"xgboost_model_{run_id}")

    print(f"Best iteration: {best_iter}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  PRAUC    : {ap:.4f}")
    print(f"  LogLoss  : {logloss:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print(f"  BSS      : {bss:.4f}")

    print("\nTraining complete. Model saved to:", LOCAL_MODEL_DIR)


if __name__ == "__main__":
    train()
