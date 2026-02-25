"""
XGBoost inference module for MovieOracle.

Loads trained quantile regression and success classifier models,
implements three-tier cold-start handling, and returns predictions
in the same response shape as the legacy model.py.

Used by backend/server.py as the primary prediction engine when
XGBoost models are available.
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
_DEFAULT_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOOKUP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "talent_lookup.json"
)

_MODEL_FILES = [
    "revenue_model_p25.joblib",
    "revenue_model_p50.joblib",
    "revenue_model_p75.joblib",
    "success_model.joblib",
]

# Budget tier thresholds (must match build_features.py)
_BUDGET_TIERS = [
    (0, 5_000_000, 0),
    (5_000_000, 20_000_000, 1),
    (20_000_000, 75_000_000, 2),
    (75_000_000, 150_000_000, 3),
    (150_000_000, float("inf"), 4),
]

_KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama",
    "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
]

# Feature importance groupings — matches the frontend's 6-category display
_IMPORTANCE_GROUPS = {
    "Director Track Record": [
        "director_avg_roi", "director_hit_rate",
        "director_avg_log_revenue", "director_film_count",
    ],
    "Producer Strength": [
        "producer_avg_log_revenue", "producer_hit_rate", "producer_film_count",
    ],
    "Writer Quality": [
        "writer_avg_log_revenue", "writer_hit_rate", "writer_film_count",
    ],
    "Budget Level": [
        "log_budget", "budget_tier",
    ],
    "Genre & Timing": [
        "genre_flag_Action", "genre_flag_Comedy", "genre_flag_Drama",
        "genre_flag_Horror", "genre_flag_Sci-Fi", "genre_flag_Thriller",
        "genre_flag_Animation", "genre_flag_Romance", "genre_flag_Fantasy",
        "genre_flag_Adventure", "is_sequel", "release_month", "release_year",
        "genre_year_avg_log_revenue", "competition_count",
    ],
    "Talent Synergy": [],  # No explicit synergy feature — shows 0%
}


def _get_budget_tier(budget: float) -> int:
    for low, high, tier in _BUDGET_TIERS:
        if low <= budget < high:
            return tier
    return 4


class MoviePredictorXGB:
    """
    XGBoost-based movie revenue and success predictor.

    Loads four trained models (p25, p50, p75 quantile regressors +
    binary success classifier) and the talent lookup JSON.

    Thread-safe for singleton use in the HTTP server.
    """

    def __init__(
        self,
        models_dir: str = _DEFAULT_MODELS_DIR,
        lookup_path: str = _DEFAULT_LOOKUP_PATH,
    ):
        """Load models and talent lookup from disk."""
        # Load models
        self.p25 = joblib.load(os.path.join(models_dir, "revenue_model_p25.joblib"))
        self.p50 = joblib.load(os.path.join(models_dir, "revenue_model_p50.joblib"))
        self.p75 = joblib.load(os.path.join(models_dir, "revenue_model_p75.joblib"))
        self.clf = joblib.load(os.path.join(models_dir, "success_model.joblib"))

        # Load feature columns (order must match training)
        cols_path = os.path.join(models_dir, "feature_columns.json")
        with open(cols_path) as f:
            self.feature_columns = json.load(f)

        # Load talent lookup
        with open(lookup_path) as f:
            self.talent_lookup = json.load(f)

        # Cache feature importances (computed once from p50 model)
        self._feature_importances = self.p50.feature_importances_

        print(f"  MoviePredictorXGB loaded: {len(self.talent_lookup.get('directors', {})):,} directors, "
              f"{len(self.talent_lookup.get('producers', {})):,} producers, "
              f"{len(self.talent_lookup.get('writers', {})):,} writers.")

    @classmethod
    def is_available(cls, models_dir: str = _DEFAULT_MODELS_DIR) -> bool:
        """Check whether all four required model files exist."""
        return all(
            os.path.exists(os.path.join(models_dir, f))
            for f in _MODEL_FILES
        ) and os.path.exists(os.path.join(models_dir, "feature_columns.json"))

    def _get_cold_start_features(self, genre: str, budget_tier: int) -> dict:
        """
        Return median features for the given (genre, budget_tier) cell.
        Falls back to global_medians if the cell is not in the lookup.
        """
        key = f"{genre}_{budget_tier}"
        medians = self.talent_lookup.get("cold_start_medians", {})
        if key in medians:
            return medians[key]
        return self.talent_lookup.get("global_medians", {
            "avg_log_revenue": 17.0,
            "avg_roi": 0.5,
            "hit_rate": 0.45,
            "film_count": 0,
        })

    def _get_talent_features(
        self, name: str, role: str, genre: str, budget_tier: int
    ) -> tuple:
        """
        Look up talent features from the lookup table and apply cold-start logic.

        Returns (features_dict, tier) where tier is 1, 2, or 3.

        Tier 1: 3+ prior films → use actual features
        Tier 2: 1–2 prior films → 50/50 blend with cold-start medians
        Tier 3: 0 prior films → use cold-start medians
        """
        role_lookup = self.talent_lookup.get(role, {})
        person_data = role_lookup.get(name)
        cold_start = self._get_cold_start_features(genre, budget_tier)

        if person_data is None:
            # Tier 3: completely unknown
            return cold_start, 3

        film_count = person_data.get("film_count", 0)

        if film_count == 0:
            return cold_start, 3

        actual = {
            "avg_log_revenue": person_data.get("avg_log_revenue", cold_start["avg_log_revenue"]),
            "avg_roi": person_data.get("avg_roi", cold_start["avg_roi"]),
            "hit_rate": person_data.get("hit_rate", cold_start["hit_rate"]),
            "film_count": film_count,
        }

        if film_count >= 3:
            # Tier 1: full history
            return actual, 1
        else:
            # Tier 2: blend 50/50 with cold-start (in feature space)
            blended = {
                k: (actual[k] + cold_start[k]) / 2.0
                for k in ["avg_log_revenue", "avg_roi", "hit_rate"]
            }
            blended["film_count"] = film_count
            return blended, 2

    def _assemble_feature_vector(self, input_data: dict) -> tuple:
        """
        Build a numpy feature vector in FEATURE_COLUMNS order from raw input.

        Returns (feature_vector, confidence_str, tier_info)
        where tier_info = (director_tier, producer_tier, writer_tier).
        """
        director = input_data.get("director", "Unknown")
        producer = input_data.get("producer", "Unknown")
        writer = input_data.get("writer", "Unknown")
        genre = input_data.get("genre", "Drama")
        budget = float(input_data.get("budget", 20_000_000))
        release_month = int(input_data.get("release_month", 6))
        is_sequel = int(input_data.get("is_sequel", 0))

        budget_tier = _get_budget_tier(budget)

        # Get talent features with cold-start handling
        dir_features, dir_tier = self._get_talent_features(
            director, "directors", genre, budget_tier
        )
        prod_features, prod_tier = self._get_talent_features(
            producer, "producers", genre, budget_tier
        )
        writer_features, writer_tier = self._get_talent_features(
            writer, "writers", genre, budget_tier
        )

        # Determine confidence level
        tiers = [dir_tier, prod_tier, writer_tier]
        if all(t == 1 for t in tiers):
            confidence = "High"
        elif any(t == 3 for t in tiers):
            confidence = "Low"
        else:
            confidence = "Medium"

        # Use current year for prospective predictions
        release_year = datetime.now().year

        # Market context proxy: use cold_start_medians for the genre
        cold_start = self._get_cold_start_features(genre, budget_tier)
        genre_year_avg_log_revenue = cold_start.get("avg_log_revenue", 17.0)
        competition_count = 5  # Reasonable average; real-time lookup not available

        # Build feature values in exact FEATURE_COLUMNS order
        feature_map = {
            "director_avg_roi": dir_features.get("avg_roi", 0.5),
            "director_hit_rate": dir_features.get("hit_rate", 0.45),
            "director_avg_log_revenue": dir_features.get("avg_log_revenue", 17.0),
            "director_film_count": float(dir_features.get("film_count", 0)),
            "producer_avg_log_revenue": prod_features.get("avg_log_revenue", 17.0),
            "producer_hit_rate": prod_features.get("hit_rate", 0.45),
            "producer_film_count": float(prod_features.get("film_count", 0)),
            "writer_avg_log_revenue": writer_features.get("avg_log_revenue", 17.0),
            "writer_hit_rate": writer_features.get("hit_rate", 0.45),
            "writer_film_count": float(writer_features.get("film_count", 0)),
            "log_budget": float(np.log1p(budget)),
            "budget_tier": float(budget_tier),
            "is_sequel": float(is_sequel),
            "release_month": float(release_month),
            "release_year": float(release_year),
            "genre_year_avg_log_revenue": float(genre_year_avg_log_revenue),
            "competition_count": float(competition_count),
        }

        # Genre one-hot flags
        for g in _KNOWN_GENRES:
            feature_map[f"genre_flag_{g}"] = 1.0 if g == genre else 0.0

        # Build vector in FEATURE_COLUMNS order
        feature_vector = np.array(
            [feature_map[col] for col in self.feature_columns],
            dtype=float
        ).reshape(1, -1)

        return feature_vector, confidence, (dir_tier, prod_tier, writer_tier)

    def _compute_feature_importance(self, feature_vector: np.ndarray) -> dict:
        """
        Compute grouped feature importances using the p50 model's gain-based
        feature_importances_ attribute.

        Returns dict matching the frontend's 6-group display format.
        """
        importances = self._feature_importances  # shape: (n_features,)
        total = importances.sum() + 1e-8

        # Build column-name to importance mapping
        col_importance = {
            col: float(importances[i]) / total * 100
            for i, col in enumerate(self.feature_columns)
        }

        # Group into 6 categories
        grouped = {}
        for group_name, feat_list in _IMPORTANCE_GROUPS.items():
            group_score = sum(col_importance.get(f, 0.0) for f in feat_list)
            grouped[group_name] = round(group_score, 1)

        # Normalize to sum to 100
        total_grouped = sum(grouped.values()) + 1e-8
        if abs(total_grouped - 100) > 5:
            grouped = {k: round(v / total_grouped * 100, 1) for k, v in grouped.items()}

        return grouped

    def predict(self, input_data: dict) -> dict:
        """
        Make a prediction for a new film.

        input_data keys:
          director (str), producer (str), writer (str),
          genre (str), budget (float), release_month (int),
          is_sequel (int, optional)

        Returns dict matching the legacy model.py predict() shape:
          {predicted_revenue, revenue_low, revenue_high, roi_percent,
           success_probability, feature_importance, confidence, budget}
        """
        X, confidence, tiers = self._assemble_feature_vector(input_data)
        budget = float(input_data.get("budget", 20_000_000))

        # Revenue predictions (back-transform from log space)
        log_p25 = float(self.p25.predict(X)[0])
        log_p50 = float(self.p50.predict(X)[0])
        log_p75 = float(self.p75.predict(X)[0])

        # Clamp quantile crossing (p25 must be ≤ p50 ≤ p75)
        log_p25 = min(log_p25, log_p50)
        log_p75 = max(log_p75, log_p50)

        revenue_low = float(np.expm1(log_p25))
        revenue_mid = float(np.expm1(log_p50))
        revenue_high = float(np.expm1(log_p75))

        # Success probability
        success_prob = float(self.clf.predict_proba(X)[0][1]) * 100

        # ROI
        roi = (revenue_mid - budget) / budget * 100 if budget > 0 else 0.0

        # Feature importance
        feature_importance = self._compute_feature_importance(X)

        return {
            "predicted_revenue": max(0, round(revenue_mid)),
            "revenue_low": max(0, round(revenue_low)),
            "revenue_high": max(0, round(revenue_high)),
            "roi_percent": round(roi, 1),
            "success_probability": round(success_prob, 1),
            "feature_importance": feature_importance,
            "confidence": confidence,
            "budget": budget,
        }
