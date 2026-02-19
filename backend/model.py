"""
ML prediction model using numpy-based linear regression with polynomial features.
No external ML library dependencies required.
"""
import numpy as np
import pandas as pd
import json
import os


class MoviePredictor:
    """Predicts movie financial performance using regularized linear regression."""

    def __init__(self):
        self.revenue_weights = None
        self.revenue_bias = None
        self.success_weights = None
        self.success_bias = None
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        self.target_std = None
        self.feature_names = None
        self.residual_std = None  # for confidence intervals

    def _extract_features(self, row):
        """Convert a data row into a feature vector."""
        features = {
            "director_avg_roi": row.get("director_avg_roi", 0.8),
            "director_hit_rate": row.get("director_hit_rate", 0.35),
            "producer_rev_mult": row.get("producer_rev_mult", 1.2),
            "producer_hit_rate": row.get("producer_hit_rate", 0.38),
            "writer_quality": row.get("writer_quality", 0.50),
            "writer_hit_rate": row.get("writer_hit_rate", 0.35),
            "log_budget": np.log1p(row.get("budget", 20_000_000)),
            "genre_rev_mult": row.get("genre_rev_mult", 1.0),
            "genre_base_success": row.get("genre_base_success", 0.50),
            "month_modifier": row.get("month_modifier", 1.0),
            # Interaction features
            "director_x_budget": row.get("director_avg_roi", 0.8) * np.log1p(row.get("budget", 20_000_000)),
            "producer_x_genre": row.get("producer_rev_mult", 1.2) * row.get("genre_rev_mult", 1.0),
            "talent_combo": (
                row.get("director_avg_roi", 0.8) *
                row.get("producer_rev_mult", 1.2) *
                row.get("writer_quality", 0.50)
            ),
            "director_roi_sq": row.get("director_avg_roi", 0.8) ** 2,
            "budget_sq": np.log1p(row.get("budget", 20_000_000)) ** 2,
        }
        return features

    def _df_to_features(self, df):
        """Convert DataFrame to feature matrix."""
        feature_dicts = [self._extract_features(row) for _, row in df.iterrows()]
        feature_df = pd.DataFrame(feature_dicts)
        self.feature_names = list(feature_df.columns)
        return feature_df.values

    def _normalize(self, X, fit=False):
        """Standardize features."""
        if fit:
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0) + 1e-8
        return (X - self.feature_means) / self.feature_stds

    def _ridge_regression(self, X, y, alpha=1.0):
        """Closed-form ridge regression: w = (X^T X + αI)^{-1} X^T y"""
        n, d = X.shape
        # Add bias column
        X_b = np.column_stack([np.ones(n), X])
        I = np.eye(d + 1)
        I[0, 0] = 0  # Don't regularize bias
        w = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y)
        return w[0], w[1:]  # bias, weights

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def _logistic_regression(self, X, y, lr=0.01, epochs=1000, alpha=0.1):
        """Gradient descent logistic regression with L2 regularization."""
        n, d = X.shape
        X_b = np.column_stack([np.ones(n), X])
        w = np.zeros(d + 1)

        for _ in range(epochs):
            z = X_b @ w
            p = self._sigmoid(z)
            grad = X_b.T @ (p - y) / n + alpha * np.concatenate([[0], w[1:]])
            w -= lr * grad

        return w[0], w[1:]  # bias, weights

    def train(self, csv_path="data/movies.csv"):
        """Train both revenue and success models."""
        print("Loading data...")
        df = pd.read_csv(csv_path)
        df = df[df["budget"] > 0].copy()

        print(f"Training on {len(df)} films...")

        # Extract features
        X = self._df_to_features(df)
        X_norm = self._normalize(X, fit=True)

        # --- Revenue model (ridge regression on log revenue) ---
        y_rev = np.log1p(df["revenue"].values)
        self.target_mean = y_rev.mean()
        self.target_std = y_rev.std()
        y_rev_norm = (y_rev - self.target_mean) / self.target_std

        self.revenue_bias, self.revenue_weights = self._ridge_regression(X_norm, y_rev_norm, alpha=2.0)

        # Compute residual std for confidence intervals
        preds = X_norm @ self.revenue_weights + self.revenue_bias
        self.residual_std = np.std(y_rev_norm - preds)

        rev_preds_actual = np.expm1(preds * self.target_std + self.target_mean)
        rev_actual = df["revenue"].values
        mape = np.mean(np.abs(rev_preds_actual - rev_actual) / (rev_actual + 1)) * 100
        r2 = 1 - np.sum((rev_actual - rev_preds_actual)**2) / np.sum((rev_actual - rev_actual.mean())**2)
        print(f"  Revenue model - MAPE: {mape:.1f}%, R²: {r2:.3f}")

        # --- Success model (logistic regression) ---
        y_success = df["success"].values.astype(float)
        self.success_bias, self.success_weights = self._logistic_regression(
            X_norm, y_success, lr=0.05, epochs=2000, alpha=0.5
        )

        success_preds = self._sigmoid(X_norm @ self.success_weights + self.success_bias)
        accuracy = np.mean((success_preds > 0.5) == y_success)
        print(f"  Success model - Accuracy: {accuracy:.1%}")

        print("Training complete!")
        return self

    def predict(self, input_data):
        """
        Make a prediction for a new film.

        input_data: dict with keys like director_avg_roi, producer_rev_mult, etc.
        Returns: dict with revenue, roi, success_prob, confidence, feature_importance
        """
        features = self._extract_features(input_data)
        x = np.array([features[k] for k in self.feature_names])
        x_norm = (x - self.feature_means) / self.feature_stds

        # Revenue prediction
        rev_pred_norm = x_norm @ self.revenue_weights + self.revenue_bias
        rev_pred_log = rev_pred_norm * self.target_std + self.target_mean
        predicted_revenue = float(np.expm1(rev_pred_log))

        # Confidence interval (using residual std)
        low_log = (rev_pred_norm - 1.5 * self.residual_std) * self.target_std + self.target_mean
        high_log = (rev_pred_norm + 1.5 * self.residual_std) * self.target_std + self.target_mean
        revenue_low = float(np.expm1(low_log))
        revenue_high = float(np.expm1(high_log))

        # Budget and ROI
        budget = input_data.get("budget", 20_000_000)
        roi = (predicted_revenue - budget) / budget if budget > 0 else 0

        # Success probability
        success_logit = x_norm @ self.success_weights + self.success_bias
        success_prob = float(self._sigmoid(success_logit))

        # Feature importance (contribution of each feature)
        contributions = np.abs(x_norm * self.revenue_weights)
        total = contributions.sum() + 1e-8
        importance = {}
        # Group into interpretable categories
        groups = {
            "Director Track Record": ["director_avg_roi", "director_hit_rate", "director_roi_sq"],
            "Producer Strength": ["producer_rev_mult", "producer_hit_rate", "producer_x_genre"],
            "Writer Quality": ["writer_quality", "writer_hit_rate"],
            "Budget Level": ["log_budget", "budget_sq"],
            "Genre & Timing": ["genre_rev_mult", "genre_base_success", "month_modifier"],
            "Talent Synergy": ["director_x_budget", "talent_combo"],
        }
        for group_name, feat_list in groups.items():
            group_contrib = sum(
                contributions[i] for i, fname in enumerate(self.feature_names) if fname in feat_list
            )
            importance[group_name] = round(float(group_contrib / total) * 100, 1)

        # Determine confidence level
        known_count = sum([
            input_data.get("director_avg_roi", 0.8) != 0.8,
            input_data.get("producer_rev_mult", 1.2) != 1.2,
            input_data.get("writer_quality", 0.5) != 0.5,
        ])
        confidence = ["Low", "Medium", "High"][min(known_count, 2)]

        return {
            "predicted_revenue": max(0, round(predicted_revenue)),
            "revenue_low": max(0, round(revenue_low)),
            "revenue_high": max(0, round(revenue_high)),
            "roi_percent": round(roi * 100, 1),
            "success_probability": round(success_prob * 100, 1),
            "feature_importance": importance,
            "confidence": confidence,
            "budget": budget,
        }

    def save(self, path="data/model.json"):
        """Save model parameters to JSON."""
        model_data = {
            "revenue_weights": self.revenue_weights.tolist(),
            "revenue_bias": float(self.revenue_bias),
            "success_weights": self.success_weights.tolist(),
            "success_bias": float(self.success_bias),
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "residual_std": float(self.residual_std),
            "feature_names": self.feature_names,
        }
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {path}")

    def load(self, path="data/model.json"):
        """Load model parameters from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.revenue_weights = np.array(data["revenue_weights"])
        self.revenue_bias = data["revenue_bias"]
        self.success_weights = np.array(data["success_weights"])
        self.success_bias = data["success_bias"]
        self.feature_means = np.array(data["feature_means"])
        self.feature_stds = np.array(data["feature_stds"])
        self.target_mean = data["target_mean"]
        self.target_std = data["target_std"]
        self.residual_std = data["residual_std"]
        self.feature_names = data["feature_names"]
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    model = MoviePredictor()
    model.train("data/movies.csv")
    model.save("data/model.json")

    # Test prediction
    test_input = {
        "director_avg_roi": 3.8,     # Christopher Nolan
        "director_hit_rate": 0.90,
        "producer_rev_mult": 3.5,    # Emma Thomas
        "producer_hit_rate": 0.88,
        "writer_quality": 0.88,      # Christopher Nolan (writer)
        "writer_hit_rate": 0.85,
        "budget": 200_000_000,
        "genre_rev_mult": 1.3,       # Sci-Fi
        "genre_base_success": 0.50,
        "month_modifier": 1.25,      # June release
    }
    result = model.predict(test_input)
    print("\n--- Test Prediction (Nolan Sci-Fi, $200M budget, June) ---")
    print(f"Revenue: ${result['predicted_revenue']:,.0f}")
    print(f"Range: ${result['revenue_low']:,.0f} - ${result['revenue_high']:,.0f}")
    print(f"ROI: {result['roi_percent']:.1f}%")
    print(f"Success Prob: {result['success_probability']:.1f}%")
    print(f"Confidence: {result['confidence']}")
    print(f"Importance: {json.dumps(result['feature_importance'], indent=2)}")
