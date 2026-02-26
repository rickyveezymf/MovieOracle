"""
XGBoost model training pipeline for MovieOracle.

Trains:
  - Three XGBoost quantile regression models (p25, p50, p75) for revenue prediction
  - One XGBoost binary classifier for success probability

Uses Optuna for hyperparameter tuning and strict temporal train/val/test splits.

Usage:
    python models/train.py

Requires:
    data/processed/features.csv (from build_features.py)

Outputs:
    models/revenue_model_p25.joblib
    models/revenue_model_p50.joblib
    models/revenue_model_p75.joblib
    models/success_model.joblib
    models/feature_columns.json
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
    print("ERROR: xgboost not installed. Run: pip install 'xgboost>=1.7.0'")
    sys.exit(1)

# Check XGBoost version for quantile support
_xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
if _xgb_version < (1, 7):
    print(f"ERROR: XGBoost >= 1.7.0 required for quantile regression. Found: {xgb.__version__}")
    print("Run: pip install 'xgboost>=1.7.0'")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Skipping hyperparameter tuning.")
    print("  Run: pip install optuna  (recommended for better accuracy)")

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODELS_DIR = SCRIPT_DIR  # Save models alongside this script

# ─── Feature columns (must match build_features.py FEATURE_COLUMNS) ──────────
FEATURE_COLUMNS = [
    "director_avg_roi", "director_hit_rate", "director_avg_log_revenue", "director_film_count",
    "producer_avg_log_revenue", "producer_hit_rate", "producer_film_count",
    "writer_avg_log_revenue", "writer_hit_rate", "writer_film_count",
    "log_budget", "budget_tier",
    "genre_flag_Action", "genre_flag_Comedy", "genre_flag_Drama", "genre_flag_Horror",
    "genre_flag_Sci-Fi", "genre_flag_Thriller", "genre_flag_Animation",
    "genre_flag_Romance", "genre_flag_Fantasy", "genre_flag_Adventure",
    "is_sequel", "release_month", "release_year",
    "genre_year_avg_log_revenue", "competition_count",
]

TARGET_REVENUE = "log_revenue"
TARGET_SUCCESS = "success"

# Temporal split thresholds
TRAIN_END = "2014-01-01"
VAL_END = "2016-01-01"

# Optuna tuning settings
N_OPTUNA_TRIALS = 20
EARLY_STOPPING_ROUNDS = 50

# Default hyperparameters (used when Optuna unavailable or as fallback)
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load features.csv and parse release_date."""
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Run python data/build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])

    # Ensure all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing feature columns in CSV: {missing}")
        sys.exit(1)

    return df


def temporal_split(df: pd.DataFrame):
    """
    Split dataset by release_date into train/val/test.
    Returns (train_df, val_df, test_df).
    Falls back to a chronological percentage split if date thresholds
    produce an empty validation set.
    """
    train_df = df[df["release_date"] < TRAIN_END].copy()
    val_df = df[(df["release_date"] >= TRAIN_END) & (df["release_date"] < VAL_END)].copy()
    test_df = df[df["release_date"] >= VAL_END].copy()

    # Fallback: if validation set is empty, split chronologically by percentage
    if len(val_df) == 0:
        print("  WARNING: Date thresholds produced empty val set. "
              "Falling back to chronological percentage split.")
        df_sorted = df.sort_values("release_date").reset_index(drop=True)
        n = len(df_sorted)
        train_end_idx = int(n * 0.70)
        val_end_idx = int(n * 0.85)
        train_df = df_sorted.iloc[:train_end_idx].copy()
        val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_sorted.iloc[val_end_idx:].copy()

    print(f"  Train: {len(train_df):,} films (before {TRAIN_END})")
    print(f"  Val:   {len(val_df):,} films ({TRAIN_END} – {VAL_END})")
    print(f"  Test:  {len(test_df):,} films (after {VAL_END}) [held out]")

    if len(train_df) < 500:
        print(f"WARNING: Training set is very small ({len(train_df)} rows). "
              "Predictions may be unreliable.")

    return train_df, val_df, test_df


def extract_arrays(df: pd.DataFrame):
    """Extract feature matrix and target arrays from a DataFrame split."""
    X = df[FEATURE_COLUMNS].values.astype(float)
    y_revenue = df[TARGET_REVENUE].values.astype(float)
    y_success = df[TARGET_SUCCESS].values.astype(float)
    return X, y_revenue, y_success


# ─── MAPE helper ─────────────────────────────────────────────────────────────

def compute_mape(y_log_actual: np.ndarray, y_log_pred: np.ndarray) -> float:
    """Compute MAPE after back-transforming from log space."""
    y_actual = np.expm1(y_log_actual)
    y_pred = np.expm1(y_log_pred)
    # Avoid division by zero
    mask = y_actual > 0
    return float(np.mean(np.abs(y_pred[mask] - y_actual[mask]) / y_actual[mask]) * 100)


# ─── Optuna hyperparameter tuning ─────────────────────────────────────────────

def optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """
    Optuna objective function for XGBoost p50 quantile model.
    Returns MAPE on validation set (lower is better).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    }

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="mae",
        verbosity=0,
        random_state=42,
        **params,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    val_preds = model.predict(X_val)
    return compute_mape(y_val, val_preds)


def run_hyperparameter_tuning(X_train, y_train, X_val, y_val) -> dict:
    """
    Run Optuna hyperparameter search.
    Returns best params dict.
    """
    if not OPTUNA_AVAILABLE:
        print("  Using default hyperparameters (install optuna for tuning).")
        return DEFAULT_PARAMS.copy()

    print(f"  Running Optuna tuning ({N_OPTUNA_TRIALS} trials) ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_params
    best_mape = study.best_value
    print(f"  Best MAPE (val): {best_mape:.1f}%")
    print(f"  Best params: {best}")
    return best


# ─── Model training ───────────────────────────────────────────────────────────

def train_quantile_model(
    X_train, y_train, X_val, y_val,
    quantile: float,
    params: dict,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost quantile regression model.
    quantile: 0.25, 0.50, or 0.75
    """
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="mae",
        verbosity=0,
        random_state=42,
        **params,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    return model


def train_success_model(
    X_train, y_train_cls, X_val, y_val_cls,
    params: dict,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary success classifier.
    Adjusts scale_pos_weight for class imbalance.
    """
    n_pos = y_train_cls.sum()
    n_neg = len(y_train_cls) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="auc",
        verbosity=0,
        random_state=42,
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 3),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train_cls,
            eval_set=[(X_val, y_val_cls)],
            verbose=False,
        )

    return model


# ─── Model saving ────────────────────────────────────────────────────────────

def save_models(p25, p50, p75, clf, output_dir: str = MODELS_DIR):
    """Save trained models and feature columns list."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(p25, os.path.join(output_dir, "revenue_model_p25.joblib"))
    joblib.dump(p50, os.path.join(output_dir, "revenue_model_p50.joblib"))
    joblib.dump(p75, os.path.join(output_dir, "revenue_model_p75.joblib"))
    joblib.dump(clf, os.path.join(output_dir, "success_model.joblib"))

    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"  Models saved to: {output_dir}/")
    print(f"    revenue_model_p25.joblib")
    print(f"    revenue_model_p50.joblib")
    print(f"    revenue_model_p75.joblib")
    print(f"    success_model.joblib")
    print(f"    feature_columns.json")


# ─── Sanity check ────────────────────────────────────────────────────────────

def sanity_check_quantiles(p25, p50, p75, X_val):
    """Verify p25 < p50 < p75 for the majority of validation predictions."""
    preds_p25 = np.expm1(p25.predict(X_val))
    preds_p50 = np.expm1(p50.predict(X_val))
    preds_p75 = np.expm1(p75.predict(X_val))

    crossing_low = np.mean(preds_p25 > preds_p50) * 100
    crossing_high = np.mean(preds_p75 < preds_p50) * 100

    if crossing_low > 10:
        print(f"  WARNING: p25 > p50 for {crossing_low:.1f}% of predictions (quantile crossing).")
        print("  This may indicate insufficient data or poor hyperparameters.")
    else:
        print(f"  Quantile ordering OK: p25 < p50 < p75 for {100 - crossing_low:.1f}% of predictions.")

    return preds_p25, preds_p50, preds_p75


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MovieOracle XGBoost Training Pipeline")
    print("=" * 60)

    # ── Load data ──
    print("\nLoading features ...")
    df = load_features()
    print(f"  Loaded {len(df):,} films.")

    print("\nSplitting data temporally ...")
    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train_rev, y_train_cls = extract_arrays(train_df)
    X_val, y_val_rev, y_val_cls = extract_arrays(val_df)
    X_test, y_test_rev, y_test_cls = extract_arrays(test_df)

    print(f"\nFeature matrix shape: {X_train.shape}")

    # ── Hyperparameter tuning ──
    print("\nHyperparameter tuning (p50 model) ...")
    best_params = run_hyperparameter_tuning(X_train, y_train_rev, X_val, y_val_rev)

    # ── Train quantile models ──
    print("\nTraining quantile revenue models ...")
    print("  Training p25 model ...")
    p25 = train_quantile_model(X_train, y_train_rev, X_val, y_val_rev, 0.25, best_params)
    print("  Training p50 model ...")
    p50 = train_quantile_model(X_train, y_train_rev, X_val, y_val_rev, 0.50, best_params)
    print("  Training p75 model ...")
    p75 = train_quantile_model(X_train, y_train_rev, X_val, y_val_rev, 0.75, best_params)

    # ── Train success classifier ──
    print("\nTraining success classifier ...")
    clf = train_success_model(X_train, y_train_cls, X_val, y_val_cls, best_params)
    print(f"  Success rate in training set: {y_train_cls.mean():.1%}")

    # ── Sanity checks ──
    if len(X_val) > 0:
        print("\nSanity check: quantile ordering ...")
        sanity_check_quantiles(p25, p50, p75, X_val)

        # ── Quick validation metrics ──
        print("\nQuick validation metrics (p50 model):")
        val_preds = p50.predict(X_val)
        val_mape = compute_mape(y_val_rev, val_preds)
        val_r2 = 1 - np.sum((y_val_rev - val_preds) ** 2) / np.sum((y_val_rev - y_val_rev.mean()) ** 2)
        print(f"  MAPE: {val_mape:.1f}%  (target: < 40%)")
        print(f"  R²:   {val_r2:.3f}  (target: > 0.55)")

        val_cls_proba = clf.predict_proba(X_val)[:, 1]
        val_cls_pred = (val_cls_proba > 0.5).astype(int)
        val_accuracy = np.mean(val_cls_pred == y_val_cls)
        print(f"\nSuccess classifier accuracy (val): {val_accuracy:.1%}")
    else:
        print("\nNo validation films available — skipping sanity checks and val metrics.")

    # ── Save models ──
    print("\nSaving models ...")
    save_models(p25, p50, p75, clf)

    # ── Test set preview ──
    if len(test_df) > 0:
        print(f"\nTest set preview ({len(test_df)} films, not used in training):")
        test_preds = p50.predict(X_test)
        test_mape = compute_mape(y_test_rev, test_preds)
        print(f"  MAPE: {test_mape:.1f}%")
    else:
        print("\nNo test films available (all films before 2023).")

    print("\nTraining complete!")
    print("Next step: python models/evaluate.py")
    print("Then start the server: bash start.sh")


if __name__ == "__main__":
    main()
