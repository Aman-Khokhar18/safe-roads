import pandas as pd
import json
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

    TEST_SIZE = config["TEST_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]

    # Only used locally now
    LOCAL_MODEL_DIR = Path(config.get("LOCAL_MODEL_DIR", "artifacts/catboost_model"))

    mlflow.set_tracking_uri("http://localhost:5000")

    # --- Data ---
    with tqdm(total=2, desc="Loading data") as p:
        pos = data_loader("roads_features_collision")
        p.update(1)
        neg = data_loader("roads_features_negatives")
        p.update(1)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, pos, neg)

    cat_cols = [c for c in config["CATEGORICAL"] if c in X_train.columns]
    features = list(X_train.columns)
    name_to_idx = {n: i for i, n in enumerate(features)}
    cat_feature_indices = [name_to_idx[n] for n in cat_cols if n in name_to_idx]

    # ---- Model ----
    model = CatBoostClassifier(
        task_type='GPU',
        loss_function="Logloss",
        eval_metric="PRAUC",
        custom_metric=["Precision", "Recall","F1"],
        iterations=3000,
        learning_rate=0.01,
        depth=6,
        min_data_in_leaf=6,
        l2_leaf_reg=1.0,
        random_seed=RANDOM_STATE,
        use_best_model=True,
        verbose=True,
        auto_class_weights="Balanced",
    )

    # ---- MLflow ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="catboost_binary") as run:

        # Log small/metadata to MLflow
        mlflow.log_text(json.dumps(list(X_train.columns), indent=2), "features.json")
        mlflow.log_params({
            "n_rows_train": X_train.shape[0],
            "n_features": X_train.shape[1],
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "early_stopping": True,
            "learning_rate_scheduler": False,
            "cat_feature_names": cat_cols,
        })
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        best_iter = model.get_best_iteration()

        y_test_pred = model.predict(X_test)
        test_acc  = accuracy_score(y_test, y_test_pred)
        test_f1   = f1_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec  = recall_score(y_test, y_test_pred)

        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
        })

        mlflow.log_dict(model.get_all_params(), "artifacts/catboost_all_params.json")

        # ---------- Local save only ----------
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = LOCAL_MODEL_DIR / "model.cbm"
        model.save_model(model_path)

        (LOCAL_MODEL_DIR / "features.json").write_text(
            json.dumps(list(X_train.columns), indent=2)
        )
        meta = {
            "best_iteration": int(best_iter),
            "class_names": model.get_param("class_names"),
            "params": model.get_all_params(),

            # Add these so handle_nans can run at serve-time:
            "CATEGORICAL": config["CATEGORICAL"],
            "NUMERICAL":  config["NUMERICAL"],
            "BOOLEAN":    config["BOOLEAN"],

            "cat_feature_indices": cat_feature_indices,
            "categorical_sentinel": "<NA>",  
        }
        (LOCAL_MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

        mlflow.log_artifacts(str(LOCAL_MODEL_DIR), artifact_path="catboost_model_local")

    print(f"Best iteration: {best_iter}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1       : {test_f1:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall   : {test_rec:.4f}")

    print("\nTraining complete. Model saved to:", LOCAL_MODEL_DIR)


if __name__ == "__main__":
    train()
