import json
import warnings
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
import argparse, os, sys, traceback, logging

# Try to import SHAP
try:
    import shap
except Exception as e:
    shap = None
    warnings.warn(f"SHAP is not available: {e}")

# === EDITABLE CONSTANTS ===
CONFIG_PATH = "configs/train.yml"
MODEL_DIR = "artifacts/xgboost_model"          # where model_{runid}.json etc. are stored
TRACKING_URI = "http://localhost:5000"         # MLflow tracking URI
LOG_TO_MLFLOW = True                            # set False to disable MLflow logging
THRESHOLD_MODE = "auto_f1"                     # "auto_f1" or "fixed"
FIXED_THRESHOLD = 0.50                           # used when THRESHOLD_MODE == "fixed"
TOP_N_IMPORTANCE = 30                           # number of features for the bar chart
OUTPUT_DIR_OVERRIDE = None                      # e.g., "artifacts/xgb_eval_custom"; None -> default path

# SHAP-specific knobs
SHAP_SAMPLE_SIZE = 5000              # cap rows for global SHAP to keep it fast
SHAP_TOP_N = 30                      # how many features to show on beeswarm/bar
SHAP_DEPENDENCE_TOP_K = 6            # per-feature dependence plots for top-K feats
SHAP_WATERFALL_EXAMPLES = 3          # how many local explanations to save
SHAP_DO_INTERACTIONS = True          # set False to skip interaction computation
SHAP_INTERACTION_SAMPLE_SIZE = 2000  # cap rows for interaction values

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
    model_candidates = [
        p for p in model_dir.glob("model_*.json")
        if p.name.startswith("model_") and not p.name.startswith("model_meta_")
    ]
    if not model_candidates:
        return None

    model_candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for mp in model_candidates:
        run_id = mp.stem[len("model_"):]
        features_ok = (model_dir / f"features_{run_id}.json").exists()
        meta_ok = (model_dir / f"model_meta_{run_id}.json").exists()
        if features_ok and meta_ok:
            return run_id

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
    cat_cols = [c for c in config.get("CATEGORICAL", []) if c in X_train.columns]
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
            try:
                num_rounds = booster.num_boosted_rounds()   # newer API
            except Exception:
                num_rounds = len(booster.get_dump())
            end = min(int(best_iter) + 1, int(num_rounds))
            return model.predict_proba(X, iteration_range=(0, end))[:, 1]
        except TypeError:
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


# ----------------------------- SHAP helpers -----------------------------
def _get_shap_explainer(model):
    if shap is None:
        raise RuntimeError("SHAP is not installed. Please `pip install shap`.")
    try:
        explainer = shap.Explainer(model, algorithm="tree")
    except Exception:
        explainer = shap.TreeExplainer(model)
    return explainer


def _compute_shap_values(model: XGBClassifier, X: pd.DataFrame, sample_size: int):
    if shap is None:
        raise RuntimeError("SHAP is not installed. Please `pip install shap`.")

    if len(X) > sample_size:
        X_used = X.sample(n=sample_size, random_state=42)
    else:
        X_used = X

    explainer = _get_shap_explainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_vals = explainer.shap_values(X_used)

    base_values = None
    if hasattr(explainer, "expected_value"):
        base_values = explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)) and len(np.atleast_1d(base_values)) > 1:
            base_values = np.atleast_1d(base_values)[-1]

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]

    if hasattr(shap_vals, "values"):
        base_values = getattr(shap_vals, "base_values", base_values)
        shap_vals = np.array(shap_vals.values)

    return explainer, X_used, shap_vals, base_values


def _plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path, top_n: int):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path, top_n: int):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _top_features_by_mean_abs(shap_values: np.ndarray, feature_names: List[str], k: int) -> List[str]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    top_idx = order[: min(k, len(feature_names))]
    return [feature_names[i] for i in top_idx]


def _plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame, features: List[str], out_dir: Path):
    for f in features:
        plt.figure()
        shap.dependence_plot(f, shap_values, X, show=False, interaction_index="auto")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence__{f}.png", bbox_inches="tight")
        plt.close()


def _make_explanation_for_row(row_idx: int, shap_values: np.ndarray, base_value, X: pd.DataFrame):
    try:
        ex = shap.Explanation(
            values=shap_values[row_idx],
            base_values=base_value if np.ndim(base_value) == 0 else np.ravel(base_value)[-1],
            data=X.iloc[row_idx].values,
            feature_names=list(X.columns),
        )
        return ex
    except Exception:
        return None


def _plot_shap_waterfalls(
    shap_values: np.ndarray,
    base_value,
    X: pd.DataFrame,
    proba: np.ndarray,
    out_dir: Path,
    k: int,
):
    if len(proba) == 0:
        return
    order = np.argsort(proba)
    low = int(order[0])
    high = int(order[-1])
    mid = int(order[len(order) // 2])

    picks = [high, mid, low][:k]
    labels = ["top", "mid", "low"][:k]
    for idx, tag in zip(picks, labels):
        ex = _make_explanation_for_row(idx, shap_values, base_value, X)
        out = out_dir / f"shap_waterfall__{tag}_idx{idx}.png"
        plt.figure()
        try:
            if ex is not None:
                shap.plots.waterfall(ex, show=False, max_display=SHAP_TOP_N)
            else:
                contrib = shap_values[idx]
                order = np.argsort(np.abs(contrib))[-SHAP_TOP_N:]
                names = X.columns[order]
                vals = contrib[order]
                plt.barh(names, vals)
                plt.title("Local contributions (fallback)")
                plt.tight_layout()
        finally:
            plt.savefig(out, bbox_inches="tight")
            plt.close()


def _compute_and_plot_interactions(model, X: pd.DataFrame, out_dir: Path, sample_size: int):
    try:
        expl = shap.TreeExplainer(model)
        X_used = X.sample(n=min(sample_size, len(X)), random_state=42) if len(X) > sample_size else X
        inter = expl.shap_interaction_values(X_used)  # shape (n, f, f)
        if isinstance(inter, list):
            inter = inter[-1]
        mean_abs = np.mean(np.abs(inter), axis=0)
        np.fill_diagonal(mean_abs, 0.0)
        f_names = list(X.columns)
        rows = []
        for i in range(len(f_names)):
            for j in range(i + 1, len(f_names)):
                rows.append((f_names[i], f_names[j], float(mean_abs[i, j])))
        df_pairs = pd.DataFrame(rows, columns=["feature_i", "feature_j", "mean_abs_interaction"]).sort_values(
            "mean_abs_interaction", ascending=False
        )
        df_pairs.to_csv(out_dir / "shap_interactions_top_pairs.csv", index=False)

        top = df_pairs.head(20).iloc[::-1]
        plt.figure(figsize=(8, max(4, len(top) * 0.35)))
        labels = [f"{a} × {b}" for a, b in zip(top["feature_i"], top["feature_j"]) ]
        plt.barh(labels, top["mean_abs_interaction"])
        plt.xlabel("Mean |interaction|")
        plt.title("Top SHAP Interaction Pairs")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interactions_top20.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        warnings.warn(f"Could not compute SHAP interactions: {e}")


# ----------------------------- Main evaluation -----------------------------
def evaluate_shap():
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

    print(f"Evaluating (SHAP) run_id={run_id}\n  model_path={model_path}\n  best_iteration={best_iter}\n  out_dir={eval_dir}")

    # MLflow logging
    if LOG_TO_MLFLOW:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"xgb_shap_eval_{run_id}")
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

    # Standard plots
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

    # === SHAP explanations & plots ===
    if shap is not None:
        try:
            X_for_shap = X_test
            explainer, Xs, shap_vals, base_val = _compute_shap_values(model, X_for_shap, SHAP_SAMPLE_SIZE)

            # Save raw values
            np.save(eval_dir / "shap_values.npy", shap_vals)
            if base_val is not None:
                (eval_dir / "shap_base_value.txt").write_text(str(base_val))

            # Global plots
            _plot_shap_beeswarm(shap_vals, Xs, eval_dir / "shap_beeswarm.png", SHAP_TOP_N)
            _plot_shap_bar(shap_vals, Xs, eval_dir / "shap_importance_bar.png", SHAP_TOP_N)

            # Per-feature dependence (top-K by mean |SHAP|)
            top_feats = _top_features_by_mean_abs(shap_vals, list(Xs.columns), SHAP_DEPENDENCE_TOP_K)
            dep_dir = eval_dir / "shap_dependence"
            _ensure_dir(dep_dir)
            _plot_shap_dependence(shap_vals, Xs, top_feats, dep_dir)

            # Local waterfalls (pick extremes and median by prob)
            xs_probs = _predict_proba(model, Xs, best_iter)
            _plot_shap_waterfalls(shap_vals, base_val, Xs, xs_probs, eval_dir, SHAP_WATERFALL_EXAMPLES)

            # Optional: interactions
            if SHAP_DO_INTERACTIONS:
                _compute_and_plot_interactions(model, X_for_shap, eval_dir, SHAP_INTERACTION_SAMPLE_SIZE)

        except Exception as e:
            warnings.warn(f"SHAP plotting failed: {e}")
    else:
        warnings.warn("Skipping SHAP plots because SHAP is not installed.")

    # MLflow logging
    if mlflow_run is not None:
        mlflow.log_params({
            "eval_run_id": run_id,
            "threshold_mode": THRESHOLD_MODE,
            "chosen_threshold": thr,
            "num_features": len(features),
            "shap_sample_size": SHAP_SAMPLE_SIZE,
            "shap_top_n": SHAP_TOP_N,
            "shap_dependence_top_k": SHAP_DEPENDENCE_TOP_K,
            "shap_waterfall_examples": SHAP_WATERFALL_EXAMPLES,
            "shap_do_interactions": SHAP_DO_INTERACTIONS,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(cls_report, "classification_report.txt")
        mlflow.log_artifacts(str(eval_dir), artifact_path=f"shap_eval_{run_id}")
        mlflow.end_run(status="FINISHED")

    print("\n=== Classification Report ===\n" + cls_report)
    print(f"Artifacts saved to: {eval_dir}")


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model: metrics + SHAP plots")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to train.yml (default: configs/train.yml)")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory containing model_*.json and companions")
    parser.add_argument("--eval-dir", default=None, help="Override output eval directory (default based on run_id)")
    parser.add_argument("--tracking-uri", default=TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Override module-level settings
    global CONFIG_PATH, MODEL_DIR, OUTPUT_DIR_OVERRIDE, TRACKING_URI, LOG_TO_MLFLOW
    CONFIG_PATH = args.config
    MODEL_DIR = args.model_dir
    OUTPUT_DIR_OVERRIDE = args.eval_dir
    TRACKING_URI = args.tracking_uri
    if args.no_mlflow:
        LOG_TO_MLFLOW = False

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)


    try:
        evaluate_shap()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli()
