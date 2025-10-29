import json
from sklearn.metrics import (brier_score_loss, average_precision_score, roc_auc_score, log_loss)
from pathlib import Path

from prefect import flow

import mlflow
from catboost import CatBoostClassifier

from safe_roads.utils.mlutil import data_loader, prepare_data
from safe_roads.utils.config import load_config


@flow(name="Train CatBoost model")
def train():
    config = load_config("configs/train.yml")
    EXPERIMENT_NAME = config['EXPERIMENT_NAME']
    RANDOM_STATE = config['RANDOM_STATE']
    EARLY_STOPPING_ROUNDS = config['EARLY_STOPPING_ROUNDS']

    LOCAL_MODEL_DIR = Path(config.get("LOCAL_MODEL_DIR", "artifacts/catboost_model"))

    mlflow.set_tracking_uri("http://localhost:5000")

    # --- Data ---
    data = next(data_loader("combined_dataset", chunksize=None, mode="train"))

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, data)

    cat_cols = [c for c in config["CATEGORICAL"] if c in X_train.columns]
    features = list(X_train.columns)
    name_to_idx = {n: i for i, n in enumerate(features)}
    cat_feature_indices = [name_to_idx[n] for n in cat_cols if n in name_to_idx]

    # ---- Model ----
    model = CatBoostClassifier(
        task_type='GPU',
        loss_function="Logloss",
        eval_metric="BrierScore",
        custom_metric=["PRAUC","AUC","Logloss"],
        iterations=100,
        learning_rate=0.01,
        depth=7,
        random_seed=RANDOM_STATE,
        use_best_model=True,
        verbose=100,
        class_weights={0: 1.0, 1: 1.0}, 
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS, 
    )

    # ---- MLflow ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="catboost_binary") as run:
        run_id = run.info.run_id
        # Log small/metadata to MLflow
        mlflow.log_text(json.dumps(list(X_train.columns), indent=2), "features.json")
        params = model.get_params()
        mlflow.log_dict(params, f"catboost_params_{run_id}.json")        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
        )

        best_iter = model.get_best_iteration()

        # --- Probability-metrics ---
        test_probs = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, test_probs)
        auc   = roc_auc_score(y_test, test_probs)
        ap    = average_precision_score(y_test, test_probs)   # PRAUC
        logloss = log_loss(y_test, test_probs, labels=[0,1])

        pi = y_test.mean()
        bss = 1.0 - brier / (pi * (1 - pi))                   # Brier Skill Score

        mlflow.log_metrics({
            "AUC": auc,
            "PRAUC": ap,
            "LogLoss": logloss,
            "brier": brier,
            "bss": bss
        })

        # ---------- Local save----------
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = LOCAL_MODEL_DIR / f"model_{run_id}.cbm"
        model.save_model(model_path)

        (LOCAL_MODEL_DIR / f"features_{run_id}.json").write_text(
            json.dumps(list(X_train.columns), indent=2)
        )
        meta = {
            "best_iteration": int(best_iter),
            "CATEGORICAL": config["CATEGORICAL"],
            "NUMERICAL":  config["NUMERICAL"],
            "BOOLEAN":    config["BOOLEAN"],
            "cat_feature_indices": cat_feature_indices,
            "categorical_sentinel": "__MISSING__", 
        }
        (LOCAL_MODEL_DIR / f"model_meta_{run_id}.json").write_text(json.dumps(meta, indent=2))

        mlflow.log_artifacts(str(LOCAL_MODEL_DIR), artifact_path=f"catboost_model_{run_id}")

    print(f"Best iteration: {best_iter}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  PRAUC    : {ap:.4f}")
    print(f"  LogLoss  : {logloss:.4f}")
    print(f"  Brier    : {brier:.4f}")  
    print(f"  BSS      : {bss:.4f}") 

    print("\nTraining complete. Model saved to:", LOCAL_MODEL_DIR)


if __name__ == "__main__":
    train()
