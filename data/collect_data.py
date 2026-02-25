"""
Data collection pipeline for MovieOracle ML training.

Downloads the TMDB movies dataset from Kaggle, enriches each film with
director/producer/writer credits from the TMDb API, and writes a clean
CSV ready for feature engineering.

Usage:
    export TMDB_API_KEY=your_key_here
    python data/collect_data.py

    # Or the script will prompt for the API key interactively.

Kaggle dataset (download manually if kaggle CLI unavailable):
    https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
    Place movies_metadata.csv in data/raw/
"""
import ast
import csv
import getpass
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed")
CACHE_PATH = os.path.join(RAW_DIR, "tmdb_cache.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "films_enriched.csv")

TMDB_CREDITS_URL = "https://api.themoviedb.org/3/movie/{movie_id}/credits"
TMDB_API_SLEEP = 0.26  # Stay well under 40 req/10s rate limit
CACHE_FLUSH_EVERY = 200  # Save cache to disk every N new fetches

# Filtering criteria
MIN_BUDGET = 100_000
MIN_REVENUE = 100_000
MIN_YEAR = 2000
MAX_YEAR = 2025

# Genre name normalization mapping
GENRE_NORMALIZE = {
    "Science Fiction": "Sci-Fi",
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Thriller",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Horror": "Horror",
    "Music": "Drama",
    "Mystery": "Thriller",
    "Romance": "Romance",
    "Thriller": "Thriller",
    "War": "Action",
    "Western": "Action",
    "Family": "Animation",
    "History": "Drama",
    "Documentary": None,  # Exclude documentaries per spec
    "TV Movie": None,     # Exclude TV movies
}

KNOWN_GENRES = {
    "Action", "Adventure", "Animation", "Comedy", "Drama",
    "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
}


# ─── Kaggle Download ──────────────────────────────────────────────────────────

def try_kaggle_download(dest_dir: str) -> bool:
    """
    Attempt to download the TMDB movies dataset via kaggle CLI.
    Returns True if successful, False otherwise.
    """
    # Check if movies_metadata.csv already exists
    if os.path.exists(os.path.join(dest_dir, "movies_metadata.csv")):
        print("  Found existing movies_metadata.csv, skipping download.")
        return True

    # Check if kaggle CLI is available
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    print("  Attempting Kaggle download: rounakbanik/the-movies-dataset ...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "rounakbanik/the-movies-dataset",
             "--unzip", "-p", dest_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("  Download complete.")
            return True
        else:
            print(f"  Kaggle download failed: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("  Download timed out.")
        return False


def print_manual_instructions():
    """Print instructions for manually downloading the dataset."""
    print("\n" + "=" * 60)
    print("MANUAL DATASET DOWNLOAD REQUIRED")
    print("=" * 60)
    print("\n1. Go to: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
    print("2. Click 'Download' (requires free Kaggle account)")
    print("3. Extract the zip file")
    print(f"4. Place 'movies_metadata.csv' in: {RAW_DIR}/")
    print("\nAlternative dataset (if the above is unavailable):")
    print("  https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies")
    print("  Place the CSV in data/raw/ and rename to movies_metadata.csv")
    print("\nThen re-run this script.")
    print("=" * 60 + "\n")


# ─── CSV Loading & Filtering ──────────────────────────────────────────────────

def parse_genre_string(genres_str: str) -> str | None:
    """
    Parse the genres column from the Kaggle dataset.
    Format is a Python list-of-dicts string like:
      "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    Returns the first recognized/normalized genre name, or None.
    """
    if not genres_str or genres_str in ("nan", "[]", ""):
        return None
    try:
        genres_list = ast.literal_eval(genres_str)
        for g in genres_list:
            name = g.get("name", "")
            normalized = GENRE_NORMALIZE.get(name)
            if normalized in KNOWN_GENRES:
                return normalized
        return None
    except (ValueError, SyntaxError):
        return None


def load_kaggle_csv(raw_dir: str) -> pd.DataFrame:
    """
    Load and filter the Kaggle movies_metadata.csv.
    Handles the rounakbanik dataset format.
    Returns filtered DataFrame with columns:
      tmdb_id, title, budget, revenue, release_date, primary_genre
    """
    path = os.path.join(raw_dir, "movies_metadata.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print_manual_instructions()
        sys.exit(1)

    print(f"  Loading {path} ...")

    # The rounakbanik dataset has some malformed rows; use error_bad_lines=False
    df = pd.read_csv(
        path,
        usecols=["id", "title", "budget", "revenue", "release_date",
                 "genres", "original_language", "status"],
        on_bad_lines="skip",
        dtype=str,
        low_memory=False
    )
    print(f"  Raw rows: {len(df):,}")

    # Cast numeric fields
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0).astype(float)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).astype(float)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    # Drop rows with invalid IDs
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # Parse genres
    df["primary_genre"] = df["genres"].apply(parse_genre_string)

    # Parse release_date and extract year
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    release_year = df["release_date"].dt.year

    # Apply filters
    mask = (
        (df["budget"] >= MIN_BUDGET) &
        (df["revenue"] >= MIN_REVENUE) &
        (release_year >= MIN_YEAR) &
        (release_year <= MAX_YEAR) &
        (df["original_language"] == "en") &
        (df["status"].isin(["Released", "nan", None]) | df["status"].isna()) &
        df["primary_genre"].notna()
    )
    df = df[mask].copy()
    print(f"  After filters (budget≥{MIN_BUDGET:,}, revenue≥{MIN_REVENUE:,}, "
          f"English, 2000-{MAX_YEAR}): {len(df):,} films")

    # Rename and select final columns
    df = df.rename(columns={"id": "tmdb_id"})
    df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")
    return df[["tmdb_id", "title", "budget", "revenue", "release_date", "primary_genre"]].reset_index(drop=True)


# ─── TMDb API Enrichment ──────────────────────────────────────────────────────

def load_cache(cache_path: str) -> dict:
    """Load existing API response cache from disk."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: str):
    """Flush cache to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def fetch_tmdb_credits(tmdb_id: int, api_key: str, cache: dict) -> dict:
    """
    Fetch director/producer/writer credits for a film from TMDb API.
    Uses local cache to avoid redundant calls.

    Returns dict: {director, primary_producer, writer}
    Falls back to 'Unknown' values on any error.
    """
    cache_key = str(tmdb_id)
    if cache_key in cache:
        return cache[cache_key]

    default = {"director": "Unknown", "primary_producer": "Unknown", "writer": "Unknown"}

    try:
        url = TMDB_CREDITS_URL.format(movie_id=tmdb_id)
        resp = requests.get(url, params={"api_key": api_key}, timeout=10)

        if resp.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(5)
            resp = requests.get(url, params={"api_key": api_key}, timeout=10)

        if resp.status_code != 200:
            cache[cache_key] = default
            return default

        data = resp.json()
        crew = data.get("crew", [])

        # Extract director (first director by credit order)
        directors = [c for c in crew if c.get("job") == "Director"]
        director = directors[0]["name"] if directors else "Unknown"

        # Extract writer (first screenplay/story credit)
        writer_jobs = {"Screenplay", "Writer", "Story", "Screenstory"}
        writers = [c for c in crew if c.get("job") in writer_jobs or
                   c.get("department") == "Writing"]
        writer = writers[0]["name"] if writers else "Unknown"

        # Extract primary producer (first executive or regular producer)
        producers = [c for c in crew if c.get("job") == "Producer"]
        if not producers:
            producers = [c for c in crew if c.get("job") == "Executive Producer"]
        primary_producer = producers[0]["name"] if producers else "Unknown"

        result = {
            "director": director,
            "primary_producer": primary_producer,
            "writer": writer,
        }
        cache[cache_key] = result
        return result

    except Exception:
        cache[cache_key] = default
        return default


def enrich_with_credits(df: pd.DataFrame, api_key: str, cache_path: str) -> pd.DataFrame:
    """
    Enrich each film with director/producer/writer credits via TMDb API.
    Uses local cache and rate-limits API calls.
    """
    cache = load_cache(cache_path)
    print(f"  Cache has {len(cache):,} entries. Fetching remaining credits ...")

    director_list = []
    producer_list = []
    writer_list = []
    new_fetches = 0

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  Progress: {i + 1:,}/{total:,} films ({new_fetches} new API calls)")

        tmdb_id = int(row["tmdb_id"])
        cache_key = str(tmdb_id)
        needs_fetch = cache_key not in cache

        credits = fetch_tmdb_credits(tmdb_id, api_key, cache)

        if needs_fetch:
            new_fetches += 1
            time.sleep(TMDB_API_SLEEP)

            if new_fetches % CACHE_FLUSH_EVERY == 0:
                save_cache(cache, cache_path)
                print(f"  Cache saved ({len(cache):,} entries)")

        director_list.append(credits["director"])
        producer_list.append(credits["primary_producer"])
        writer_list.append(credits["writer"])

    # Final cache save
    save_cache(cache, cache_path)
    print(f"  Credits fetched. Total API calls: {new_fetches:,}")

    df = df.copy()
    df["director"] = director_list
    df["primary_producer"] = producer_list
    df["writer"] = writer_list

    # Drop films with no director credit
    before = len(df)
    df = df[df["director"] != "Unknown"].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} films with no director credit.")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Get TMDb API key ──
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        print("TMDb API key not found in TMDB_API_KEY environment variable.")
        api_key = getpass.getpass("Enter your TMDb API key: ").strip()
    if not api_key:
        print("ERROR: TMDb API key is required.")
        sys.exit(1)

    # Validate key with a quick test call
    test_resp = requests.get(
        "https://api.themoviedb.org/3/movie/550/credits",
        params={"api_key": api_key}, timeout=10
    )
    if test_resp.status_code != 200:
        print(f"ERROR: TMDb API key is invalid or network is unavailable. Status: {test_resp.status_code}")
        sys.exit(1)
    print("TMDb API key validated.")

    # ── Step 1: Download / locate Kaggle data ──
    print("\nStep 1: Locating Kaggle dataset ...")
    if not try_kaggle_download(RAW_DIR):
        if not os.path.exists(os.path.join(RAW_DIR, "movies_metadata.csv")):
            print_manual_instructions()
            sys.exit(1)

    # ── Step 2: Load and filter ──
    print("\nStep 2: Loading and filtering dataset ...")
    df = load_kaggle_csv(RAW_DIR)

    # ── Step 3: Enrich with TMDb credits ──
    print(f"\nStep 3: Enriching {len(df):,} films with TMDb credits ...")
    print(f"  This may take {len(df) * TMDB_API_SLEEP / 60:.0f}–{len(df) * TMDB_API_SLEEP * 2 / 60:.0f} minutes on first run.")
    print(f"  Subsequent runs use cache at: {CACHE_PATH}")
    df = enrich_with_credits(df, api_key, CACHE_PATH)

    # ── Step 4: Final cleanup ──
    df = df.dropna(subset=["tmdb_id", "title", "budget", "revenue",
                            "release_date", "primary_genre",
                            "director", "primary_producer", "writer"])
    df = df.drop_duplicates(subset=["tmdb_id"])

    # ── Step 5: Write output ──
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['release_date'].min()} — {df['release_date'].max()}")
    print(f"  Budget median: ${df['budget'].median():,.0f}")
    print(f"  Revenue median: ${df['revenue'].median():,.0f}")
    print(f"  Genres: {sorted(df['primary_genre'].value_counts().to_dict().items(), key=lambda x: -x[1])}")
    print(f"\nNext step: python data/build_features.py")


if __name__ == "__main__":
    main()
