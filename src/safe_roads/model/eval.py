import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from catboost import CatBoostClassifier
import mlflow

# Same utilities as train
from safe_roads.utils.mlutil import data_loader, prepare_data
from safe_roads.utils.config import load_config


def ensure_features_order(df: pd.DataFrame, features_json_path: Path) -> pd.DataFrame:
    with features_json_path.open("r", encoding="utf-8") as f:
        train_features = json.load(f)
    for col in train_features:
        if col not in df.columns:
            df[col] = pd.NA
    return df[train_features]


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    pr_path = out_dir / "pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight", dpi=150)
    plt.close()

    return roc_path, pr_path, roc_auc, ap


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    return cm_path


def main():
    # ---- Configs / paths (reuse training config & artifacts) ----
    cfg_path = os.getenv("TRAIN_CONFIG_PATH", "configs/train.yml")
    config = load_config(cfg_path)

    local_model_dir = Path(config.get("LOCAL_MODEL_DIR", "artifacts/catboost_model"))
    model_path = local_model_dir / "model.cbm"
    features_json = local_model_dir / "features.json"
    eval_dir = Path(config.get("EVAL_DIR", "artifacts/eval"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not features_json.exists():
        raise FileNotFoundError(f"features.json not found at {features_json}")

    # ---- Build the SAME splits as training ----
    pos = data_loader("roads_features_collision")
    neg = data_loader("roads_features_negatives")
    if pos is None or len(pos) == 0:
        raise RuntimeError("No positive samples from 'roads_features_collision'.")
    if neg is None or len(neg) == 0:
        raise RuntimeError("No negative samples from 'roads_features_negatives'.")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, pos, neg)

    # Align test features to training order
    X_test = ensure_features_order(X_test.copy(), features_json)

    # ---- Load model ----
    model = CatBoostClassifier()
    model.load_model(model_path)

    # ---- Predict ----
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # positive-class probability

    # ---- Metrics ----
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    metrics = {
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "n_test": int(len(y_test)),
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- Plots ----
    roc_path, pr_path, _, _ = plot_roc_pr_curves(y_test, y_prob, eval_dir)
    cm_path = plot_confusion(y_test, y_pred, eval_dir)

    # ---- MLflow logging ----
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(config.get("EXPERIMENT_NAME", "default"))

    with mlflow.start_run(run_name="catboost_eval"):
        # Params related to evaluation context (non-PII, small)
        mlflow.log_params({
            "eval_split": "test",
            "n_test": len(y_test),
            "model_path": str(model_path),
            "features_json": str(features_json),
        })
        # Metrics
        mlflow.log_metrics(metrics)
        # Artifacts
        mlflow.log_artifact(str(roc_path), artifact_path="eval")
        mlflow.log_artifact(str(pr_path), artifact_path="eval")
        mlflow.log_artifact(str(cm_path), artifact_path="eval")
        mlflow.log_artifact(str(eval_dir / "metrics.json"), artifact_path="eval")

        preview = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        prev_path = eval_dir / "predictions_preview.csv"
        preview.to_csv(prev_path, index=False)
        mlflow.log_artifact(str(prev_path), artifact_path="eval")

    # ---- Console summary ----
    print("=== Evaluation (Test Split) ===")
    print(f"Samples        : {len(y_test)}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"F1             : {f1:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"ROC AUC        : {roc_auc:.4f}")
    print(f"Avg Precision  : {ap:.4f}")
    print(f"Experiment     : {config.get('EXPERIMENT_NAME', 'default')}")
    print(f"Artifacts in   : {eval_dir}")


if __name__ == "__main__":
    main()
