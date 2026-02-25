"""
Evaluation suite for MovieOracle XGBoost models.

Runs comprehensive metrics on the held-out test set and performs
temporal validation (train pre-2020, test 2020-2025).

Usage:
    python models/evaluate.py

Requires:
    data/processed/features.csv   (from build_features.py)
    models/*.joblib               (from train.py)
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# ─── Dependency checks ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed.")
    sys.exit(1)

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, brier_score_loss
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed. Some metrics will be skipped.")
    print("  Run: pip install scikit-learn")

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODELS_DIR = SCRIPT_DIR
REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.txt")

TRAIN_END = "2020-01-01"
VAL_END = "2023-01-01"


# ─── Data utilities ───────────────────────────────────────────────────────────

def load_features_and_split():
    """Load features and return (train_df, val_df, test_df, feature_columns)."""
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: {FEATURES_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])

    # Load feature columns from models dir
    cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    if not os.path.exists(cols_path):
        print(f"ERROR: {cols_path} not found. Run train.py first.")
        sys.exit(1)
    with open(cols_path) as f:
        feature_columns = json.load(f)

    train_df = df[df["release_date"] < TRAIN_END]
    val_df = df[(df["release_date"] >= TRAIN_END) & (df["release_date"] < VAL_END)]
    test_df = df[df["release_date"] >= VAL_END]

    return train_df, val_df, test_df, feature_columns


def df_to_arrays(df: pd.DataFrame, feature_columns: list):
    """Extract X, y_log, y_raw, y_cls arrays from a DataFrame."""
    X = df[feature_columns].values.astype(float)
    y_log = df["log_revenue"].values.astype(float)
    y_raw = np.expm1(y_log)
    y_cls = df["success"].values.astype(float)
    return X, y_log, y_raw, y_cls


def load_models():
    """Load all four trained models from disk."""
    required = ["revenue_model_p25.joblib", "revenue_model_p50.joblib",
                "revenue_model_p75.joblib", "success_model.joblib"]
    for fname in required:
        fpath = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: Model file not found: {fpath}")
            print("Run python models/train.py first.")
            sys.exit(1)

    p25 = joblib.load(os.path.join(MODELS_DIR, "revenue_model_p25.joblib"))
    p50 = joblib.load(os.path.join(MODELS_DIR, "revenue_model_p50.joblib"))
    p75 = joblib.load(os.path.join(MODELS_DIR, "revenue_model_p75.joblib"))
    clf = joblib.load(os.path.join(MODELS_DIR, "success_model.joblib"))
    return p25, p50, p75, clf


# ─── Revenue model evaluation ────────────────────────────────────────────────

def compute_mape(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in percentage."""
    mask = y_actual > 0
    return float(np.mean(np.abs(y_pred[mask] - y_actual[mask]) / y_actual[mask]) * 100)


def evaluate_revenue_models(
    p25, p50, p75,
    X: np.ndarray,
    y_log: np.ndarray,
    y_raw: np.ndarray,
    split_name: str = "test",
) -> dict:
    """
    Evaluate revenue model performance.

    Metrics:
      MAPE: Mean absolute percentage error (in revenue space)
      R²:   Coefficient of determination (in log space)
      IQR coverage: fraction of actuals within [p25_pred, p75_pred]
                    (target ~50% for a well-calibrated IQR)
      MedAE: Median absolute error in $M
    """
    p25_log = p25.predict(X)
    p50_log = p50.predict(X)
    p75_log = p75.predict(X)

    # Back-transform
    p25_raw = np.expm1(p25_log)
    p50_raw = np.expm1(p50_log)
    p75_raw = np.expm1(p75_log)

    # MAPE (using p50 as central estimate)
    mape = compute_mape(y_raw, p50_raw)

    # R² in log space
    ss_res = np.sum((y_log - p50_log) ** 2)
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # IQR coverage: what fraction of actuals falls within [p25, p75]?
    in_iqr = np.mean((y_raw >= p25_raw) & (y_raw <= p75_raw)) * 100

    # Median absolute error in $M
    med_ae = float(np.median(np.abs(p50_raw - y_raw))) / 1_000_000

    return {
        "mape": round(mape, 1),
        "r2": round(float(r2), 3),
        "iqr_coverage": round(float(in_iqr), 1),
        "median_abs_error_M": round(med_ae, 1),
        "n_samples": len(y_raw),
        "split": split_name,
    }


# ─── Success classifier evaluation ───────────────────────────────────────────

def evaluate_success_model(clf, X: np.ndarray, y_cls: np.ndarray, split_name: str = "test") -> dict:
    """Evaluate success classifier with AUC-ROC, accuracy, precision, recall, F1, Brier score."""
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)

    result = {
        "accuracy": round(float(np.mean(pred == y_cls)) * 100, 1),
        "n_samples": len(y_cls),
        "success_rate": round(float(y_cls.mean()) * 100, 1),
        "split": split_name,
    }

    if SKLEARN_AVAILABLE:
        result["auc_roc"] = round(float(roc_auc_score(y_cls, proba)), 3)
        result["precision"] = round(float(precision_score(y_cls, pred, zero_division=0)), 3)
        result["recall"] = round(float(recall_score(y_cls, pred, zero_division=0)), 3)
        result["f1"] = round(float(f1_score(y_cls, pred, zero_division=0)), 3)
        result["brier_score"] = round(float(brier_score_loss(y_cls, proba)), 4)
    else:
        result["note"] = "Install scikit-learn for AUC-ROC and other metrics"

    return result


# ─── Temporal validation ──────────────────────────────────────────────────────

def temporal_validation(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Re-train a simplified XGBoost model on pre-2020 data and test on 2020-2025.
    Does NOT re-run Optuna — uses fixed parameters for speed.
    Simulates the real-world scenario where you predict future films.
    """
    print("  Running temporal validation (train < 2020, test 2020-2025) ...")

    pre2020 = df[df["release_date"] < "2020-01-01"]
    post2020 = df[(df["release_date"] >= "2020-01-01") & (df["release_date"] < "2025-01-01")]

    if len(pre2020) < 100 or len(post2020) < 10:
        return {"error": f"Insufficient data: {len(pre2020)} pre-2020, {len(post2020)} post-2020"}

    X_pre = pre2020[feature_columns].values.astype(float)
    y_pre_log = pre2020["log_revenue"].values.astype(float)
    y_pre_cls = pre2020["success"].values.astype(float)

    X_post = post2020[feature_columns].values.astype(float)
    y_post_log = post2020["log_revenue"].values.astype(float)
    y_post_raw = np.expm1(y_post_log)
    y_post_cls = post2020["success"].values.astype(float)

    # Simplified params (no Optuna re-run)
    simple_params = dict(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rev_model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.5, **simple_params
        )
        rev_model.fit(X_pre, y_pre_log)

        clf_model = xgb.XGBClassifier(objective="binary:logistic", **simple_params)
        clf_model.fit(X_pre, y_pre_cls)

    p50_post = np.expm1(rev_model.predict(X_post))
    mape = compute_mape(y_post_raw, p50_post)
    r2 = 1 - np.sum((y_post_log - rev_model.predict(X_post)) ** 2) / np.sum(
        (y_post_log - y_post_log.mean()) ** 2
    )

    result = {
        "train_size": len(pre2020),
        "test_size": len(post2020),
        "mape": round(mape, 1),
        "r2": round(float(r2), 3),
    }

    if SKLEARN_AVAILABLE:
        proba = clf_model.predict_proba(X_post)[:, 1]
        result["auc_roc"] = round(float(roc_auc_score(y_post_cls, proba)), 3)

    print(f"  Temporal validation: MAPE={mape:.1f}%, R²={r2:.3f}, "
          f"train={len(pre2020)}, test={len(post2020)}")
    return result


# ─── Report formatting ────────────────────────────────────────────────────────

def format_report(rev_metrics: dict, cls_metrics: dict, temporal: dict) -> str:
    """Format evaluation results into a human-readable report."""
    lines = [
        "=" * 60,
        "MovieOracle Model Evaluation Report",
        "=" * 60,
        "",
        f"Data split: {rev_metrics.get('split', 'test').upper()}  "
        f"({rev_metrics['n_samples']} films)",
        "",
        "── Revenue Model (p50 = central estimate) ──────────────",
        f"  MAPE:                {rev_metrics['mape']:.1f}%",
        f"  R²:                  {rev_metrics['r2']:.3f}",
        f"  IQR Coverage:        {rev_metrics['iqr_coverage']:.1f}%  (target: ~50%)",
        f"  Median Abs Error:    ${rev_metrics['median_abs_error_M']:.1f}M",
        "",
        "Expected ranges for box office prediction:",
        "  MAPE:      25-40% = strong   40-60% = acceptable   >60% = needs work",
        "  R²:        >0.55 = good      >0.70 = very good     >0.80 = check for leakage",
        "  IQR Cov:   ~50% = well-calibrated (p25-p75 interval captures correct fraction)",
        "",
        "── Success Classifier ───────────────────────────────────",
        f"  Accuracy:            {cls_metrics['accuracy']:.1f}%",
        f"  Success rate in set: {cls_metrics['success_rate']:.1f}%",
    ]

    if "auc_roc" in cls_metrics:
        lines += [
            f"  AUC-ROC:             {cls_metrics['auc_roc']:.3f}  (target: 0.70-0.80)",
            f"  Precision:           {cls_metrics['precision']:.3f}",
            f"  Recall:              {cls_metrics['recall']:.3f}",
            f"  F1 Score:            {cls_metrics['f1']:.3f}",
            f"  Brier Score:         {cls_metrics['brier_score']:.4f}  (0=perfect, 0.25=random)",
        ]

    lines += [
        "",
        "Expected ranges:",
        "  AUC-ROC:   >0.70 = good   >0.75 = strong   >0.80 = very strong",
        "",
    ]

    if "error" in temporal:
        lines.append(f"── Temporal Validation ──────────────────────────────────")
        lines.append(f"  Skipped: {temporal['error']}")
    else:
        lines += [
            "── Temporal Validation (train<2020 → test 2020-2025) ────",
            f"  Train size:          {temporal['train_size']:,}",
            f"  Test size:           {temporal['test_size']:,}",
            f"  MAPE:                {temporal['mape']:.1f}%",
            f"  R²:                  {temporal['r2']:.3f}",
        ]
        if "auc_roc" in temporal:
            lines.append(f"  AUC-ROC:             {temporal['auc_roc']:.3f}")
        lines.append("  (Higher MAPE expected vs. random split — this is the honest estimate)")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading features ...")
    train_df, val_df, test_df, feature_columns = load_features_and_split()

    df_full = pd.concat([train_df, val_df, test_df])
    print(f"  Full dataset: {len(df_full):,} films")
    print(f"  Test set (after 2023): {len(test_df):,} films")

    print("\nLoading models ...")
    p25, p50, p75, clf = load_models()

    # ── Decide which split to evaluate on ──
    if len(test_df) >= 20:
        eval_df = test_df
        split_name = "test"
        print(f"\nEvaluating on held-out test set ({len(test_df)} films) ...")
    elif len(val_df) >= 20:
        eval_df = val_df
        split_name = "validation"
        print(f"\nTest set too small; evaluating on validation set ({len(val_df)} films) ...")
    else:
        eval_df = pd.concat([val_df, test_df])
        split_name = "val+test"
        print(f"\nUsing combined val+test set ({len(eval_df)} films) ...")

    X_eval, y_log_eval, y_raw_eval, y_cls_eval = df_to_arrays(eval_df, feature_columns)

    # ── Revenue evaluation ──
    rev_metrics = evaluate_revenue_models(p25, p50, p75, X_eval, y_log_eval, y_raw_eval, split_name)

    # ── Classifier evaluation ──
    cls_metrics = evaluate_success_model(clf, X_eval, y_cls_eval, split_name)

    # ── Temporal validation ──
    temporal = temporal_validation(df_full, feature_columns)

    # ── Format and print report ──
    report = format_report(rev_metrics, cls_metrics, temporal)
    print("\n" + report)

    # ── Save report to file ──
    with open(REPORT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
