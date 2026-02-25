"""
Feature engineering pipeline for MovieOracle ML training.

Reads films_enriched.csv, engineers all predictive features using strict
temporal ordering (no data leakage), and writes:
  - data/processed/features.csv    (full feature matrix + targets)
  - data/talent_lookup.json        (pre-computed talent stats for inference)

Usage:
    python data/build_features.py

Requires:
    data/processed/films_enriched.csv (from collect_data.py)
"""
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "processed", "films_enriched.csv")
OUTPUT_FEATURES = os.path.join(SCRIPT_DIR, "processed", "features.csv")
OUTPUT_LOOKUP = os.path.join(SCRIPT_DIR, "talent_lookup.json")

KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama",
    "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
]

BUDGET_TIERS = [
    (0, 5_000_000, 0),          # Micro
    (5_000_000, 20_000_000, 1), # Low
    (20_000_000, 75_000_000, 2),# Mid
    (75_000_000, 150_000_000, 3),# High
    (150_000_000, float("inf"), 4),# Blockbuster
]

SEQUEL_PATTERN = re.compile(
    r"\b(2|3|4|5|6|7|8|9|II|III|IV|VI|VII|VIII|IX|Part|Chapter|Returns|"
    r"Rises|Again|Reloaded|Revolutions|Legacy|Resurrection|Reborn|Rise|"
    r"Forever|Strikes Back)\b",
    re.IGNORECASE
)

# Cold-start defaults used when a person has no prior history at all
# These get replaced by data-driven medians in the final talent_lookup.json
GLOBAL_DEFAULTS = {
    "avg_roi": 0.5,
    "hit_rate": 0.45,
    "avg_log_revenue": 17.0,
    "film_count": 0,
}


# ─── Helper: budget tier ─────────────────────────────────────────────────────

def get_budget_tier(budget: float) -> int:
    for low, high, tier in BUDGET_TIERS:
        if low <= budget < high:
            return tier
    return 4


# ─── Helper: talent stats from history ───────────────────────────────────────

def compute_talent_stats(history: list) -> dict:
    """
    Compute aggregate stats from a list of prior film dicts.
    Each dict has: {revenue, budget, log_revenue}

    Returns: {avg_roi, hit_rate, avg_log_revenue, film_count}
    """
    if not history:
        return GLOBAL_DEFAULTS.copy()

    rois = []
    hits = 0
    log_revs = []

    for film in history:
        rev = film["revenue"]
        bud = film["budget"]
        log_rev = film["log_revenue"]

        if bud > 0:
            roi = (rev - bud) / bud
            rois.append(roi)
        if bud > 0 and rev > bud * 1.5:
            hits += 1
        log_revs.append(log_rev)

    n = len(history)
    avg_roi = float(np.mean(rois)) if rois else 0.5
    hit_rate = hits / n if n > 0 else 0.45
    avg_log_revenue = float(np.mean(log_revs)) if log_revs else 17.0

    return {
        "avg_roi": round(avg_roi, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_log_revenue": round(avg_log_revenue, 4),
        "film_count": n,
    }


# ─── Core: rolling talent features (no data leakage) ─────────────────────────

def compute_rolling_talent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute talent track record features using only PRIOR films for each row.

    CRITICAL: df must already be sorted by release_date ascending before calling.
    The accumulator is updated AFTER each row's features are computed, so the
    film itself never influences its own features.

    Adds columns:
      director_avg_roi, director_hit_rate, director_avg_log_revenue, director_film_count
      producer_avg_log_revenue, producer_hit_rate, producer_film_count
      writer_avg_log_revenue, writer_hit_rate, writer_film_count
    """
    print("  Computing rolling talent features (temporal, no leakage) ...")

    # Running accumulators: name -> list of {revenue, budget, log_revenue}
    dir_history = {}
    prod_history = {}
    writer_history = {}

    director_features = []
    producer_features = []
    writer_features = []

    for _, row in df.iterrows():
        director = row["director"]
        producer = row["primary_producer"]
        writer = row["writer"]
        revenue = float(row["revenue"])
        budget = float(row["budget"])
        log_revenue = math.log1p(revenue)

        film_record = {"revenue": revenue, "budget": budget, "log_revenue": log_revenue}

        # Compute features from PRIOR history (before adding this film)
        dir_stats = compute_talent_stats(dir_history.get(director, []))
        prod_stats = compute_talent_stats(prod_history.get(producer, []))
        writer_stats = compute_talent_stats(writer_history.get(writer, []))

        director_features.append(dir_stats)
        producer_features.append(prod_stats)
        writer_features.append(writer_stats)

        # Now update history with this film
        dir_history.setdefault(director, []).append(film_record)
        prod_history.setdefault(producer, []).append(film_record)
        writer_history.setdefault(writer, []).append(film_record)

    # Add talent feature columns to DataFrame
    df = df.copy()
    df["director_avg_roi"] = [f["avg_roi"] for f in director_features]
    df["director_hit_rate"] = [f["hit_rate"] for f in director_features]
    df["director_avg_log_revenue"] = [f["avg_log_revenue"] for f in director_features]
    df["director_film_count"] = [f["film_count"] for f in director_features]

    df["producer_avg_log_revenue"] = [f["avg_log_revenue"] for f in producer_features]
    df["producer_hit_rate"] = [f["hit_rate"] for f in producer_features]
    df["producer_film_count"] = [f["film_count"] for f in producer_features]

    df["writer_avg_log_revenue"] = [f["avg_log_revenue"] for f in writer_features]
    df["writer_hit_rate"] = [f["hit_rate"] for f in writer_features]
    df["writer_film_count"] = [f["film_count"] for f in writer_features]

    return df, dir_history, prod_history, writer_history


# ─── Film characteristic features ────────────────────────────────────────────

def add_film_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_budget, budget_tier, genre one-hot flags, is_sequel,
    release_month, release_year.
    """
    print("  Adding film characteristic features ...")
    df = df.copy()

    df["log_budget"] = np.log1p(df["budget"].astype(float))
    df["budget_tier"] = df["budget"].astype(float).apply(get_budget_tier)

    # Genre one-hot flags
    for genre in KNOWN_GENRES:
        col = f"genre_flag_{genre}"
        df[col] = (df["primary_genre"] == genre).astype(int)

    # Sequel detection from title
    df["is_sequel"] = df["title"].apply(
        lambda t: 1 if isinstance(t, str) and SEQUEL_PATTERN.search(t) else 0
    )

    # Temporal features
    release_dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_month"] = release_dt.dt.month.fillna(6).astype(int)
    df["release_year"] = release_dt.dt.year.fillna(2010).astype(int)

    return df


# ─── Market context features ──────────────────────────────────────────────────

def compute_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market context features using only prior films:
      - genre_year_avg_log_revenue: avg log(revenue) of same-genre films in prior 3 years
      - competition_count: films of same genre released within prior 90 days

    O(N²) but N ≤ 15,000 so this completes in < 60 seconds.
    df must be sorted by release_date ascending.
    """
    print("  Computing market context features ...")
    df = df.copy()
    release_dates = pd.to_datetime(df["release_date"])
    genres = df["primary_genre"].values
    log_revenues = np.log1p(df["revenue"].astype(float).values)

    genre_year_avg_list = []
    competition_list = []

    three_years = timedelta(days=3 * 365)
    ninety_days = timedelta(days=90)

    for i in range(len(df)):
        current_date = release_dates.iloc[i]
        current_genre = genres[i]
        three_years_ago = current_date - three_years
        ninety_days_ago = current_date - ninety_days

        same_genre_recent = []
        competition = 0

        for j in range(i):  # Only prior films (j < i)
            if genres[j] != current_genre:
                continue
            past_date = release_dates.iloc[j]
            if past_date >= three_years_ago:
                same_genre_recent.append(log_revenues[j])
            if past_date >= ninety_days_ago:
                competition += 1

        genre_year_avg_list.append(
            float(np.mean(same_genre_recent)) if same_genre_recent else GLOBAL_DEFAULTS["avg_log_revenue"]
        )
        competition_list.append(competition)

    df["genre_year_avg_log_revenue"] = genre_year_avg_list
    df["competition_count"] = competition_list

    return df


# ─── Talent lookup JSON ───────────────────────────────────────────────────────

def build_talent_lookup(
    df: pd.DataFrame,
    dir_history: dict,
    prod_history: dict,
    writer_history: dict,
) -> dict:
    """
    Build the talent_lookup.json from the FINAL (post-all-films) history.
    Also computes cold_start_medians per (genre, budget_tier) cell
    and global_medians as the ultimate fallback.
    """
    print("  Building talent lookup JSON ...")

    def build_role_lookup(history: dict) -> dict:
        lookup = {}
        for name, films in history.items():
            if name == "Unknown":
                continue
            stats = compute_talent_stats(films)
            stats["name"] = name
            lookup[name] = stats
        return lookup

    directors_lookup = build_role_lookup(dir_history)
    producers_lookup = build_role_lookup(prod_history)
    writers_lookup = build_role_lookup(writer_history)

    # Compute global medians (from all films)
    all_roi = df["director_avg_roi"].values
    all_hit = df["director_hit_rate"].values
    all_log_rev = np.log1p(df["revenue"].astype(float).values)

    global_medians = {
        "avg_log_revenue": round(float(np.median(all_log_rev)), 4),
        "avg_roi": round(float(np.median(all_roi[all_roi > -10])), 4),
        "hit_rate": round(float(np.median(all_hit)), 4),
        "film_count": 0,
    }

    # Compute cold_start_medians per (genre, budget_tier)
    cold_start_medians = {}
    for genre in KNOWN_GENRES:
        genre_mask = df["primary_genre"] == genre
        genre_df = df[genre_mask]
        for tier in range(5):
            tier_mask = genre_df["budget_tier"] == tier
            tier_df = genre_df[tier_mask]
            if len(tier_df) < 5:
                continue
            tier_log_rev = np.log1p(tier_df["revenue"].astype(float).values)
            tier_roi = tier_df["director_avg_roi"].values
            tier_hit = tier_df["director_hit_rate"].values
            key = f"{genre}_{tier}"
            cold_start_medians[key] = {
                "avg_log_revenue": round(float(np.median(tier_log_rev)), 4),
                "avg_roi": round(float(np.median(tier_roi[tier_roi > -10])), 4),
                "hit_rate": round(float(np.median(tier_hit)), 4),
                "film_count": 0,
            }

    print(f"  Directors in lookup: {len(directors_lookup):,}")
    print(f"  Producers in lookup: {len(producers_lookup):,}")
    print(f"  Writers in lookup: {len(writers_lookup):,}")
    print(f"  Cold-start cells: {len(cold_start_medians):,}")

    return {
        "directors": directors_lookup,
        "producers": producers_lookup,
        "writers": writers_lookup,
        "cold_start_medians": cold_start_medians,
        "global_medians": global_medians,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

# All feature columns in training order (must match models/train.py FEATURE_COLUMNS)
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


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run python data/collect_data.py first.")
        sys.exit(1)

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {len(df):,} films.")

    # Sort by release_date BEFORE any feature computation (anti-leakage)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df = df.sort_values("release_date").reset_index(drop=True)
    print(f"  Sorted by date. Range: {df['release_date'].min().date()} — {df['release_date'].max().date()}")

    # Add target columns early (needed for talent history computation)
    df["log_revenue"] = np.log1p(df["revenue"].astype(float))
    df["success"] = ((df["revenue"].astype(float)) > (df["budget"].astype(float) * 1.5)).astype(int)

    # Feature engineering (in temporal order)
    df, dir_history, prod_history, writer_history = compute_rolling_talent_features(df)
    df = add_film_characteristics(df)
    df = compute_market_context(df)

    # Restore release_date as string for the CSV
    df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")

    # Final cleanup: drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["log_revenue", "success"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with NaN features.")

    # ── Correlation check ──
    print("\nFeature correlations with log(revenue):")
    for col in ["director_avg_log_revenue", "log_budget", "producer_hit_rate", "director_hit_rate"]:
        if col in df.columns:
            corr = df[col].corr(df["log_revenue"])
            print(f"  {col}: {corr:.3f}")

    # ── Write features CSV ──
    output_cols = ["tmdb_id", "title", "release_date", "budget", "revenue",
                   "primary_genre", "director", "primary_producer", "writer"] + \
                  FEATURE_COLUMNS + ["log_revenue", "success"]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(OUTPUT_FEATURES, index=False)
    print(f"\nFeatures written to: {OUTPUT_FEATURES}")
    print(f"  Shape: {df[output_cols].shape}")
    print(f"  Success rate: {df['success'].mean():.1%}")

    # ── Build and write talent lookup ──
    lookup = build_talent_lookup(df, dir_history, prod_history, writer_history)
    with open(OUTPUT_LOOKUP, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"\nTalent lookup written to: {OUTPUT_LOOKUP}")
    print(f"\nNext step: python models/train.py")


if __name__ == "__main__":
    main()
