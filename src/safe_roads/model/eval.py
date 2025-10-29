import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

import mlflow
from xgboost import XGBClassifier

# === EDITABLE CONSTANTS ===
CONFIG_PATH = "configs/train.yml"
MODEL_DIR = "artifacts/xgboost_model"          # where model_{runid}.json etc. are stored
TRACKING_URI = "http://localhost:5000"         # MLflow tracking URI
LOG_TO_MLFLOW = True                           # set False to disable MLflow logging
THRESHOLD_MODE = "auto_f1"                     # "auto_f1" or "fixed"
FIXED_THRESHOLD = 0.50                          # used when THRESHOLD_MODE == "fixed"
TOP_N_IMPORTANCE = 30                           # number of features for the bar chart
OUTPUT_DIR_OVERRIDE = None                     # e.g., "artifacts/xgb_eval_custom"; None -> default path

# Project utilities
from safe_roads.utils.mlutil import data_loader, prepare_data
from safe_roads.utils.config import load_config


# ----------------------------- Helpers -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_latest_run_id(model_dir: Path) -> Optional[str]:
    """
    Return latest run_id based on the mtime of *model_<runid>.json* files only.
    Ensures that features_<runid>.json and model_meta_<runid>.json also exist.
    """
    # Only consider real model files, not metadata
    model_candidates = [
        p for p in model_dir.glob("model_*.json")
        if p.name.startswith("model_") and not p.name.startswith("model_meta_")
    ]
    if not model_candidates:
        return None

    # Sort newest first
    model_candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Pick the newest model file whose companions also exist
    for mp in model_candidates:
        run_id = mp.stem[len("model_"):]
        features_ok = (model_dir / f"features_{run_id}.json").exists()
        meta_ok = (model_dir / f"model_meta_{run_id}.json").exists()
        if features_ok and meta_ok:
            return run_id

    # Fallback: return newest model even if companions are missing
    return model_candidates[0].stem[len("model_"):]



def _load_artifacts(model_dir: Path, run_id: str) -> Tuple[Path, List[str], Dict]:
    model_path = model_dir / f"model_{run_id}.json"
    features_path = model_dir / f"features_{run_id}.json"
    meta_path = model_dir / f"model_meta_{run_id}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    features = json.loads(features_path.read_text())
    meta = json.loads(meta_path.read_text())
    return model_path, features, meta


def _prepare_dataset(config: Dict) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    data = next(data_loader("combined_dataset", chunksize=None, mode="train"))
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, data)

    # align categorical dtypes like in training
    cat_cols = [c for c in config["CATEGORICAL"] if c in X_train.columns]
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X_train[c]):
            X_train[c] = X_train[c].astype("category")
            X_val[c] = X_val[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    return X_train, y_train, X_val, y_val, X_test, y_test, cat_cols


def _load_model(model_path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def _predict_proba(model, X, best_iter):
    # Use early-stopped iteration if available
    if best_iter is not None:
        try:
            booster = model.get_booster()
            # Exclusive end; ensure we don't exceed available rounds
            try:
                num_rounds = booster.num_boosted_rounds()   # newer API
            except Exception:
                # Fallback: count trees in dump (works for gbtree)
                num_rounds = len(booster.get_dump())
            end = min(int(best_iter) + 1, int(num_rounds))
            return model.predict_proba(X, iteration_range=(0, end))[:, 1]
        except TypeError:
            # Older xgboost: fall back to default behavior
            pass
    return model.predict_proba(X)[:, 1]



# ----------------------------- Metrics & Plots -----------------------------
def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    ll = log_loss(y_true, probs, labels=[0, 1])
    brier = brier_score_loss(y_true, probs)
    pi = float(np.mean(y_true))
    bss = 1.0 - brier / (pi * (1.0 - pi)) if 0.0 < pi < 1.0 else np.nan
    return {
        "AUC": float(auc),
        "PRAUC": float(ap),
        "LogLoss": float(ll),
        "Brier": float(brier),
        "BSS": float(bss),
        "Prevalence": pi,
    }


def choose_threshold(y_true: np.ndarray, probs: np.ndarray, mode: str = "auto_f1", fixed: Optional[float] = None):
    if mode == "fixed" and fixed is not None:
        t = float(fixed)
    else:
        thresholds = np.linspace(0.0, 1.0, 500)
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            y_pred = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        t = best_t

    y_pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return t, {"precision": prec, "recall": rec, "f1": f1}


def plot_roc(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration(y_true: np.ndarray, probs: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (Reliability) Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_threshold_sweep(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    thresholds = np.linspace(0.0, 1.0, 201)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    plt.figure()
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str = "Confusion Matrix") -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])  # predicted
    plt.yticks(tick_marks, ["0", "1"])  # true

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     verticalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------- Feature Importance -----------------------------
def _map_f_to_names(score_dict: Dict[str, float], feature_names: List[str]) -> pd.DataFrame:
    rows = []
    for k, v in score_dict.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            name = feature_names[idx] if idx < len(feature_names) else k
        else:
            name = k
        rows.append((name, float(v)))
    df = pd.DataFrame(rows, columns=["feature", "score"]).sort_values("score", ascending=False)
    return df


def compute_feature_importances(model: XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
    booster = model.get_booster()
    types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    frames = []
    for t in types:
        sd = booster.get_score(importance_type=t)
        df = _map_f_to_names(sd, feature_names)
        df.rename(columns={"score": t}, inplace=True)
        frames.append(df)
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="feature", how="outer")
    for c in types:
        if c in out:
            out[c] = out[c].fillna(0.0)
    out.sort_values("gain", ascending=False, inplace=True)
    return out


def plot_top_importance(df_imp: pd.DataFrame, out_path: Path, top_n: int = 30, column: str = "gain") -> None:
    df_top = df_imp.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    plt.barh(df_top["feature"], df_top[column])
    plt.xlabel(column)
    plt.title(f"Top {top_n} Feature Importance ({column})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



# ----------------------------- Main evaluation -----------------------------
def evaluate():
    # Load config & experiment name
    config = load_config(CONFIG_PATH)
    experiment_name = config["EXPERIMENT_NAME"]

    model_dir = Path(MODEL_DIR)
    run_id = _find_latest_run_id(model_dir)
    if run_id is None:
        raise RuntimeError(f"No model_* files found in {model_dir}")

    eval_dir = Path(OUTPUT_DIR_OVERRIDE) if OUTPUT_DIR_OVERRIDE else Path(f"artifacts/evals/xgboost_eval_{run_id}")
    _ensure_dir(eval_dir)

    model_path, features, meta = _load_artifacts(model_dir, run_id)
    best_iter = meta.get("best_iteration")

    print(f"Evaluating run_id={run_id}\n  model_path={model_path}\n  best_iteration={best_iter}\n  out_dir={eval_dir}")

    # MLflow logging
    if LOG_TO_MLFLOW:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"xgboost_eval_{run_id}")
    else:
        mlflow_run = None

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test, _ = _prepare_dataset(config)

    # Align columns to training order
    missing = [c for c in features if c not in X_test.columns]
    if missing:
        raise ValueError(f"Test set missing expected features: {missing[:20]} ...")
    X_test = X_test[features]

    # Model & predictions
    model = _load_model(model_path)
    test_probs = _predict_proba(model, X_test, best_iter)

    # Metrics
    metrics = compute_metrics(y_test, test_probs)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.6f}" if isinstance(v, (int, float)) else f"{k:>12}: {v}")

    # Threshold selection
    thr, thr_stats = choose_threshold(
        y_test,
        test_probs,
        mode=("fixed" if THRESHOLD_MODE == "fixed" else "auto_f1"),
        fixed=FIXED_THRESHOLD,
    )
    print(
        f"\nChosen threshold: {thr:.4f}  "
        f"(precision={thr_stats['precision']:.4f}, recall={thr_stats['recall']:.4f}, f1={thr_stats['f1']:.4f})"
    )

    y_pred = (test_probs >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)
    (eval_dir / "classification_report.txt").write_text(cls_report)

    # Save predictions table
    pd.DataFrame({"y_true": y_test, "proba": test_probs, "y_pred": y_pred}).to_csv(
        eval_dir / "predictions.csv", index=False
    )

    # Plots
    plot_roc(y_test, test_probs, eval_dir / "roc_curve.png")
    plot_pr(y_test, test_probs, eval_dir / "pr_curve.png")
    plot_calibration(y_test, test_probs, eval_dir / "calibration_curve.png")
    plot_threshold_sweep(y_test, test_probs, eval_dir / "threshold_sweep.png")
    plot_confusion_matrix(cm, eval_dir / "confusion_matrix.png")

    # Feature importances
    feat_imp = compute_feature_importances(model, features)
    feat_imp.to_csv(eval_dir / "feature_importance_all.csv", index=False)
    plot_top_importance(
        feat_imp,
        eval_dir / f"feature_importance_top{TOP_N_IMPORTANCE}_gain.png",
        top_n=TOP_N_IMPORTANCE,
        column="gain",
    )

    # MLflow logging
    if mlflow_run is not None:
        mlflow.log_params({
            "eval_run_id": run_id,
            "threshold_mode": THRESHOLD_MODE,
            "chosen_threshold": thr,
            "num_features": len(features),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(cls_report, "classification_report.txt")
        mlflow.log_artifacts(str(eval_dir), artifact_path=f"eval_{run_id}")
        mlflow.end_run(status="FINISHED")

    print("\n=== Classification Report ===\n" + cls_report)
    print(f"Artifacts saved to: {eval_dir}")


if __name__ == "__main__":
    evaluate()
