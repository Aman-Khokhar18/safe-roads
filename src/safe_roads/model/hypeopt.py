import os
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.metrics import (
    brier_score_loss, average_precision_score, roc_auc_score, log_loss
)

from prefect import flow
import mlflow
from catboost import CatBoostClassifier

# Optional: if you prefer MLflow's CatBoost flavor
# import mlflow.catboost

from safe_roads.utils.mlutil import data_loader, prepare_data
from safe_roads.utils.config import load_config


def _maybe_load_hopt_config():
    try:
        h = load_config("configs/hyperopt.yml")
    except Exception:
        h = {}
    # Added SAVE_TRIAL_MODELS knob (default False)
    return {
        "MAX_EVALS": int(h.get("MAX_EVALS", 30)),
        "DEPTH_MIN": int(h.get("DEPTH_MIN", 4)),
        "DEPTH_MAX": int(h.get("DEPTH_MAX", 10)),
        "POS_WEIGHT_LO": float(h.get("POS_WEIGHT_LO", 0.5)),
        "POS_WEIGHT_HI": float(h.get("POS_WEIGHT_HI", 10.0)),
        "TUNE_ITERATIONS": int(h.get("TUNE_ITERATIONS", 1500)),  
        "SAVE_TRIAL_MODELS": bool(h.get("SAVE_TRIAL_MODELS", False)),
    }


def tune_depth_and_class_weights(
    *,
    base_params,
    X_train, y_train,
    X_val, y_val,
    cat_cols,
    early_stopping_rounds,
    experiment_name,  
    random_state,
    max_evals,
    depth_min, depth_max,
    pos_weight_lo, pos_weight_hi,
    tune_iterations,
    save_trial_models=False,
):
    """
    Hyperopt objective: minimize validation Brier score (probability-only).
    Creates a nested MLflow run per trial with unique names/tags.
    Optionally saves each trial's CatBoost model as an artifact.
    """
    space = {
        "depth": hp.quniform("depth", depth_min, depth_max, 1),
        "w_pos": hp.loguniform("w_pos", math.log(pos_weight_lo), math.log(pos_weight_hi)),
    }
    trials = Trials()

    def _cast(p):
        return {
            "depth": int(p["depth"]),
            "class_weights": {0: 1.0, 1: float(p["w_pos"])},
        }

    def objective(p):
        suggested = _cast(p)
        params = dict(base_params)
        params.update(suggested)

        # Allow ES to find best_iter during tuning
        params["iterations"] = max(int(params.get("iterations", tune_iterations)), tune_iterations)
        params["verbose"] = False

        model = CatBoostClassifier(**params)

        # Unique trial index & naming
        trial_idx = len(trials.trials)  # 0-based
        run_name = (
            f"hopt_trial_{trial_idx:03d}"
            f"_depth{suggested['depth']}"
            f"_wpos{suggested['class_weights'][1]:.4f}"
        )

        # Start nested MLflow run for each trial
        # NOTE: Do not call mlflow.set_experiment() here; do it outside in train()
        with mlflow.start_run(run_name=run_name, nested=True):
            # Helpful tags & params for filtering/search in the UI
            mlflow.set_tags({
                "stage": "hyperopt",
                "trial_index": trial_idx,
                "tuning_objective": "brier_score",
            })
            mlflow.log_params({
                "hopt_depth": suggested["depth"],
                "hopt_class_weights": suggested["class_weights"],
                "hopt_iterations_cap": params["iterations"],
                "base_learning_rate": base_params.get("learning_rate"),
            })

            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_cols,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

            val_probs = model.predict_proba(X_val)[:, 1]
            brier = brier_score_loss(y_val, val_probs)
            auc   = roc_auc_score(y_val, val_probs)
            ap    = average_precision_score(y_val, val_probs)
            ll    = log_loss(y_val, val_probs, labels=[0, 1])

            mlflow.log_metrics({
                "val_brier": float(brier),
                "val_auc": float(auc),
                "val_prauc": float(ap),
                "val_logloss": float(ll),
                "val_best_iteration": int(model.get_best_iteration()),
            })

            # Optional: save per-trial model as artifact so each child run has its model
            if save_trial_models:
                with tempfile.TemporaryDirectory() as td:
                    tmp_path = os.path.join(td, f"model_trial_{trial_idx:03d}.cbm")
                    model.save_model(tmp_path)
                    mlflow.log_artifact(tmp_path, artifact_path="trial_model")

        return {"loss": float(brier), "status": STATUS_OK}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        rstate=np.random.default_rng(random_state),
        trials=trials,
    )
    return _cast(best)


@flow(name="Train CatBoost model (Hyperopt: depth + class_weights; probability-only)")
def train():
    # ----- Configs -----
    config = load_config("configs/train.yml")
    EXPERIMENT_NAME = config['EXPERIMENT_NAME']
    RANDOM_STATE = config['RANDOM_STATE']
    EARLY_STOPPING_ROUNDS = config['EARLY_STOPPING_ROUNDS']
    LOCAL_MODEL_DIR = Path(config.get("LOCAL_MODEL_DIR", "artifacts/catboost_model"))

    HCONF = _maybe_load_hopt_config()

    # ----- MLflow -----
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ----- Data -----
    data = next(data_loader("combined_dataset", chunksize=None))
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, data)

    cat_cols = [c for c in config["CATEGORICAL"] if c in X_train.columns]
    features = list(X_train.columns)
    name_to_idx = {n: i for i, n in enumerate(features)}
    cat_feature_indices = [name_to_idx[n] for n in cat_cols if n in name_to_idx]

    # ----- Final model baseline (match your final train script) -----
    base_params = dict(
        task_type='GPU',
        loss_function="Logloss",
        eval_metric="BrierScore",
        custom_metric=["PRAUC","AUC","Logloss"],
        iterations=300,                
        learning_rate=0.01,
        depth=6,                       # will be tuned
        random_seed=RANDOM_STATE,
        use_best_model=True,
        verbose=100,
        class_weights={0: 1.0, 1: 1.0} # will be tuned
    )

    with mlflow.start_run(run_name="catboost_binary") as run:
        run_id = run.info.run_id

        # Log features & hyperopt knobs (for reproducibility)
        mlflow.log_text(json.dumps(list(X_train.columns), indent=2), "features.json")
        mlflow.log_params({
            "HYPEROPT_MAX_EVALS": HCONF["MAX_EVALS"],
            "HYPEROPT_DEPTH_MIN": HCONF["DEPTH_MIN"],
            "HYPEROPT_DEPTH_MAX": HCONF["DEPTH_MAX"],
            "HYPEROPT_POS_WEIGHT_LO": HCONF["POS_WEIGHT_LO"],
            "HYPEROPT_POS_WEIGHT_HI": HCONF["POS_WEIGHT_HI"],
            "HYPEROPT_TUNE_ITERATIONS": HCONF["TUNE_ITERATIONS"],
            "HYPEROPT_SAVE_TRIAL_MODELS": HCONF["SAVE_TRIAL_MODELS"],
        })

        # ----- Hyperopt (probability-only objective) -----
        best = tune_depth_and_class_weights(
            base_params=base_params,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            cat_cols=cat_cols,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            experiment_name=EXPERIMENT_NAME,
            random_state=RANDOM_STATE,
            max_evals=HCONF["MAX_EVALS"],
            depth_min=HCONF["DEPTH_MIN"],
            depth_max=HCONF["DEPTH_MAX"],
            pos_weight_lo=HCONF["POS_WEIGHT_LO"],
            pos_weight_hi=HCONF["POS_WEIGHT_HI"],
            tune_iterations=HCONF["TUNE_ITERATIONS"],
            save_trial_models=HCONF["SAVE_TRIAL_MODELS"],
        )
        mlflow.log_params({
            "hopt_best_depth": best["depth"],
            "hopt_best_class_weights": best["class_weights"],
        })

        # ----- Final train with tuned params -----
        final_params = dict(base_params)
        final_params.update(best)
        model = CatBoostClassifier(**final_params)

        # Log final params (same style as your train)
        params = model.get_params()
        mlflow.log_params({
            "loss_function": params.get("loss_function"),
            "eval_metric": params.get("eval_metric"),
            "custom_metric": ",".join(params.get("custom_metric") or []),
            "iterations": params.get("iterations"),
            "learning_rate": params.get("learning_rate"),
            "depth": params.get("depth"),
            "random_seed": params.get("random_seed"),
            "class_weights": params.get("class_weights"),
        })

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        best_iter = model.get_best_iteration()

        # ----- Probability-only metrics on test -----
        test_probs = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, test_probs)
        auc   = roc_auc_score(y_test, test_probs)
        ap    = average_precision_score(y_test, test_probs)   # PRAUC
        ll    = log_loss(y_test, test_probs, labels=[0,1])

        pi = y_test.mean()
        bss = 1.0 - brier / (pi * (1 - pi))

        mlflow.log_metrics({
            "AUC": float(auc),
            "PRAUC": float(ap),
            "LogLoss": float(ll),
            "brier": float(brier),
            "bss": float(bss)
        })

        mlflow.log_dict(model.get_all_params(), "artifacts/catboost_all_params.json")

        # ----- Save artifacts (same structure as your train) -----
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = LOCAL_MODEL_DIR / f"model_{run_id}.cbm"
        model.save_model(model_path)

        (LOCAL_MODEL_DIR / f"features_{run_id}.json").write_text(
            json.dumps(list(X_train.columns), indent=2)
        )
        meta = {
            "best_iteration": int(best_iter),
            "class_names": model.get_param("class_names"),
            "params": model.get_all_params(),
            "CATEGORICAL": config["CATEGORICAL"],
            "NUMERICAL":  config["NUMERICAL"],
            "BOOLEAN":    config["BOOLEAN"],
            "cat_feature_indices": cat_feature_indices,
            "categorical_sentinel": "__MISSING__",
        }
        (LOCAL_MODEL_DIR / f"model_meta_{run_id}.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifacts(str(LOCAL_MODEL_DIR), artifact_path="catboost_model_local")

    # Console summary (probability-only)
    print(f"Best iteration: {best_iter}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  PRAUC    : {ap:.4f}")
    print(f"  LogLoss  : {ll:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print(f"  BSS      : {bss:.4f}")
    print(f"  Tuned depth={best['depth']}, class_weights={best['class_weights']}")
    print("\nTraining complete. Model saved to:", LOCAL_MODEL_DIR)


if __name__ == "__main__":
    train()
