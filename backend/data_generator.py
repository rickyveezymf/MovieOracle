"""
Generate realistic synthetic movie data for training the prediction model.
Based on real-world patterns from Box Office Mojo / TMDb data distributions.
"""
import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# ─── Real-ish talent pools ───────────────────────────────────────────────────

DIRECTORS = {
    "Christopher Nolan":     {"avg_roi": 3.8, "hit_rate": 0.90, "style": "blockbuster"},
    "Steven Spielberg":      {"avg_roi": 3.2, "hit_rate": 0.85, "style": "blockbuster"},
    "Denis Villeneuve":      {"avg_roi": 2.1, "hit_rate": 0.75, "style": "prestige"},
    "Greta Gerwig":          {"avg_roi": 3.5, "hit_rate": 0.80, "style": "prestige"},
    "Jordan Peele":          {"avg_roi": 5.0, "hit_rate": 0.90, "style": "horror_specialist"},
    "James Wan":             {"avg_roi": 4.5, "hit_rate": 0.85, "style": "horror_specialist"},
    "Taika Waititi":         {"avg_roi": 2.8, "hit_rate": 0.75, "style": "comedy"},
    "Ryan Coogler":          {"avg_roi": 4.0, "hit_rate": 0.85, "style": "blockbuster"},
    "Ari Aster":             {"avg_roi": 6.0, "hit_rate": 0.70, "style": "horror_specialist"},
    "Wes Anderson":          {"avg_roi": 1.5, "hit_rate": 0.60, "style": "indie"},
    "Martin Scorsese":       {"avg_roi": 1.8, "hit_rate": 0.65, "style": "prestige"},
    "Quentin Tarantino":     {"avg_roi": 2.5, "hit_rate": 0.80, "style": "prestige"},
    "James Cameron":         {"avg_roi": 5.5, "hit_rate": 0.95, "style": "blockbuster"},
    "Ridley Scott":          {"avg_roi": 1.6, "hit_rate": 0.55, "style": "prestige"},
    "David Fincher":         {"avg_roi": 2.0, "hit_rate": 0.70, "style": "prestige"},
    "The Russo Brothers":    {"avg_roi": 4.2, "hit_rate": 0.85, "style": "blockbuster"},
    "Michael Bay":           {"avg_roi": 2.2, "hit_rate": 0.65, "style": "blockbuster"},
    "Jon Favreau":           {"avg_roi": 3.0, "hit_rate": 0.80, "style": "blockbuster"},
    "M. Night Shyamalan":    {"avg_roi": 2.8, "hit_rate": 0.55, "style": "thriller"},
    "Chloe Zhao":            {"avg_roi": 1.2, "hit_rate": 0.50, "style": "indie"},
    "Barry Jenkins":         {"avg_roi": 1.8, "hit_rate": 0.65, "style": "indie"},
    "Damien Chazelle":       {"avg_roi": 2.5, "hit_rate": 0.70, "style": "prestige"},
    "Bong Joon-ho":          {"avg_roi": 3.0, "hit_rate": 0.75, "style": "prestige"},
    "Sam Raimi":             {"avg_roi": 2.5, "hit_rate": 0.70, "style": "blockbuster"},
    "Tim Burton":            {"avg_roi": 1.8, "hit_rate": 0.60, "style": "fantasy"},
    "Matt Reeves":           {"avg_roi": 2.8, "hit_rate": 0.75, "style": "blockbuster"},
    "Edgar Wright":          {"avg_roi": 1.5, "hit_rate": 0.65, "style": "comedy"},
    "Emerald Fennell":       {"avg_roi": 2.5, "hit_rate": 0.70, "style": "thriller"},
    "Robert Eggers":         {"avg_roi": 2.0, "hit_rate": 0.60, "style": "horror_specialist"},
    "Unknown Director":      {"avg_roi": 0.8, "hit_rate": 0.35, "style": "indie"},
}

PRODUCERS = {
    "Kevin Feige":           {"avg_rev_mult": 4.0, "hit_rate": 0.92},
    "Jerry Bruckheimer":     {"avg_rev_mult": 2.5, "hit_rate": 0.70},
    "Kathleen Kennedy":      {"avg_rev_mult": 3.0, "hit_rate": 0.75},
    "Jason Blum":            {"avg_rev_mult": 8.0, "hit_rate": 0.85},
    "Emma Thomas":           {"avg_rev_mult": 3.5, "hit_rate": 0.88},
    "David Heyman":          {"avg_rev_mult": 3.2, "hit_rate": 0.80},
    "Scott Rudin":           {"avg_rev_mult": 1.8, "hit_rate": 0.65},
    "Amy Pascal":            {"avg_rev_mult": 2.8, "hit_rate": 0.72},
    "Brad Pitt":             {"avg_rev_mult": 2.0, "hit_rate": 0.60},
    "Jordan Peele":          {"avg_rev_mult": 5.5, "hit_rate": 0.85},
    "A24":                   {"avg_rev_mult": 3.5, "hit_rate": 0.65},
    "Megan Ellison":         {"avg_rev_mult": 1.5, "hit_rate": 0.55},
    "Neal H. Moritz":        {"avg_rev_mult": 2.8, "hit_rate": 0.68},
    "Lorenzo di Bonaventura": {"avg_rev_mult": 2.2, "hit_rate": 0.60},
    "Charles Roven":         {"avg_rev_mult": 2.5, "hit_rate": 0.65},
    "Jon Landau":            {"avg_rev_mult": 5.0, "hit_rate": 0.90},
    "Frank Marshall":        {"avg_rev_mult": 2.5, "hit_rate": 0.70},
    "Peter Chernin":         {"avg_rev_mult": 2.0, "hit_rate": 0.62},
    "Mary Parent":           {"avg_rev_mult": 2.8, "hit_rate": 0.70},
    "Unknown Producer":      {"avg_rev_mult": 1.2, "hit_rate": 0.38},
}

WRITERS = {
    "Aaron Sorkin":          {"quality": 0.82, "hit_rate": 0.70},
    "Christopher Nolan":     {"quality": 0.88, "hit_rate": 0.85},
    "Quentin Tarantino":     {"quality": 0.85, "hit_rate": 0.80},
    "Greta Gerwig":          {"quality": 0.80, "hit_rate": 0.78},
    "Jordan Peele":          {"quality": 0.90, "hit_rate": 0.88},
    "Charlie Kaufman":       {"quality": 0.75, "hit_rate": 0.50},
    "Emerald Fennell":       {"quality": 0.78, "hit_rate": 0.70},
    "Bong Joon-ho":          {"quality": 0.85, "hit_rate": 0.75},
    "Taylor Sheridan":       {"quality": 0.80, "hit_rate": 0.72},
    "Tony Kushner":          {"quality": 0.78, "hit_rate": 0.65},
    "Eric Roth":             {"quality": 0.75, "hit_rate": 0.68},
    "Damien Chazelle":       {"quality": 0.82, "hit_rate": 0.72},
    "Ryan Coogler":          {"quality": 0.80, "hit_rate": 0.82},
    "Edgar Wright":          {"quality": 0.78, "hit_rate": 0.65},
    "Denis Villeneuve":      {"quality": 0.80, "hit_rate": 0.72},
    "James Gunn":            {"quality": 0.78, "hit_rate": 0.80},
    "Diablo Cody":           {"quality": 0.72, "hit_rate": 0.58},
    "Taika Waititi":         {"quality": 0.75, "hit_rate": 0.70},
    "Rian Johnson":          {"quality": 0.78, "hit_rate": 0.68},
    "Unknown Writer":        {"quality": 0.50, "hit_rate": 0.35},
}

GENRES = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Thriller", "Animation", "Romance", "Fantasy", "Adventure"]

GENRE_MODIFIERS = {
    "Action":    {"budget_mult": 1.4, "rev_mult": 1.5, "base_success": 0.55},
    "Comedy":    {"budget_mult": 0.6, "rev_mult": 0.8, "base_success": 0.50},
    "Drama":     {"budget_mult": 0.5, "rev_mult": 0.6, "base_success": 0.40},
    "Horror":    {"budget_mult": 0.3, "rev_mult": 1.8, "base_success": 0.65},
    "Sci-Fi":    {"budget_mult": 1.3, "rev_mult": 1.3, "base_success": 0.50},
    "Thriller":  {"budget_mult": 0.7, "rev_mult": 1.0, "base_success": 0.52},
    "Animation": {"budget_mult": 1.2, "rev_mult": 1.6, "base_success": 0.60},
    "Romance":   {"budget_mult": 0.4, "rev_mult": 0.7, "base_success": 0.42},
    "Fantasy":   {"budget_mult": 1.5, "rev_mult": 1.4, "base_success": 0.48},
    "Adventure": {"budget_mult": 1.3, "rev_mult": 1.5, "base_success": 0.55},
}

MONTH_MODIFIERS = {
    1: 0.70, 2: 0.72, 3: 0.82, 4: 0.85,
    5: 1.10, 6: 1.25, 7: 1.30, 8: 1.05,
    9: 0.75, 10: 0.80, 11: 1.15, 12: 1.20,
}


def generate_dataset(n_films=6000):
    """Generate a realistic synthetic movie dataset."""
    records = []
    dir_names = list(DIRECTORS.keys())
    prod_names = list(PRODUCERS.keys())
    writer_names = list(WRITERS.keys())

    for i in range(n_films):
        # Pick talent (weighted toward known names, with some unknowns)
        if np.random.random() < 0.15:
            director = "Unknown Director"
        else:
            director = np.random.choice(dir_names[:-1])
        if np.random.random() < 0.15:
            producer = "Unknown Producer"
        else:
            producer = np.random.choice(prod_names[:-1])
        if np.random.random() < 0.20:
            writer = "Unknown Writer"
        else:
            writer = np.random.choice(writer_names[:-1])

        genre = np.random.choice(GENRES)
        release_month = np.random.randint(1, 13)
        year = np.random.randint(2005, 2026)

        dir_info = DIRECTORS[director]
        prod_info = PRODUCERS[producer]
        writer_info = WRITERS[writer]
        genre_info = GENRE_MODIFIERS[genre]
        month_mod = MONTH_MODIFIERS[release_month]

        # Generate budget (log-normal distribution, influenced by genre)
        base_budget = np.exp(np.random.normal(17.5, 1.2))  # median ~40M
        budget = base_budget * genre_info["budget_mult"]
        budget = np.clip(budget, 500_000, 400_000_000)

        # Generate revenue based on talent + genre + timing
        talent_factor = (
            dir_info["avg_roi"] * 0.35 +
            prod_info["avg_rev_mult"] * 0.25 +
            writer_info["quality"] * 3.0 * 0.15 +
            genre_info["rev_mult"] * 0.15 +
            month_mod * 0.10
        )

        # Add non-linear interactions
        if dir_info["style"] == "horror_specialist" and genre == "Horror":
            talent_factor *= 1.3
        if dir_info["style"] == "blockbuster" and genre in ["Action", "Adventure", "Sci-Fi"]:
            talent_factor *= 1.2
        if dir_info["style"] == "comedy" and genre == "Comedy":
            talent_factor *= 1.15

        # Revenue = budget * talent_factor * noise
        noise = np.exp(np.random.normal(0, 0.6))
        revenue = budget * talent_factor * noise * 0.55

        # Clip revenue
        revenue = np.clip(revenue, 10_000, 3_000_000_000)

        roi = (revenue - budget) / budget
        success = 1 if revenue > budget * 1.5 else 0  # 1.5x for break-even after P&A

        records.append({
            "title": f"Film_{i+1:04d}",
            "year": year,
            "director": director,
            "producer": producer,
            "writer": writer,
            "genre": genre,
            "budget": round(budget),
            "revenue": round(revenue),
            "release_month": release_month,
            "roi": round(roi, 3),
            "success": success,
            "director_avg_roi": dir_info["avg_roi"],
            "director_hit_rate": dir_info["hit_rate"],
            "producer_rev_mult": prod_info["avg_rev_mult"],
            "producer_hit_rate": prod_info["hit_rate"],
            "writer_quality": writer_info["quality"],
            "writer_hit_rate": writer_info["hit_rate"],
            "genre_rev_mult": genre_info["rev_mult"],
            "genre_base_success": genre_info["base_success"],
            "month_modifier": month_mod,
        })

    df = pd.DataFrame(records)
    return df


def build_talent_database():
    """Export talent lists with their stats for the frontend search."""
    talent_db = {
        "directors": [
            {"name": name, "avg_roi": round(info["avg_roi"], 2), "hit_rate": round(info["hit_rate"], 2), "style": info["style"]}
            for name, info in DIRECTORS.items() if name != "Unknown Director"
        ],
        "producers": [
            {"name": name, "avg_rev_mult": round(info["avg_rev_mult"], 2), "hit_rate": round(info["hit_rate"], 2)}
            for name, info in PRODUCERS.items() if name != "Unknown Producer"
        ],
        "writers": [
            {"name": name, "quality": round(info["quality"], 2), "hit_rate": round(info["hit_rate"], 2)}
            for name, info in WRITERS.items() if name != "Unknown Writer"
        ],
        "genres": GENRES,
    }
    return talent_db


if __name__ == "__main__":
    print("Generating movie dataset...")
    df = generate_dataset(6000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/movies.csv", index=False)
    print(f"Generated {len(df)} films -> data/movies.csv")
    print(f"Success rate: {df['success'].mean():.1%}")
    print(f"Median budget: ${df['budget'].median():,.0f}")
    print(f"Median revenue: ${df['revenue'].median():,.0f}")

    talent_db = build_talent_database()
    with open("data/talent.json", "w") as f:
        json.dump(talent_db, f, indent=2)
    print("Talent database -> data/talent.json")
