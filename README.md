# 🎬 BoxOffice Oracle — Movie Financial Success Predictor

An AI-powered web application that predicts whether a movie will be financially successful by analyzing historical box office and film production data.

## Features

- **Talent Search** — Searchable dropdowns for 29 directors, 19 producers, and 19 writers with real-world track records
- **Revenue Prediction** — Estimated worldwide gross with confidence intervals
- **ROI Analysis** — Return on investment percentage with profit/loss breakdown
- **Success Probability** — Visual gauge showing likelihood of breaking even
- **Prediction Explainability** — Bar chart showing which factors (director history, budget, genre, etc.) drove the prediction
- **Budget Slider** — Interactive gradient slider from $500K to $400M with tier labels
- **Genre & Timing** — 10 genres and monthly seasonality modifiers

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/rickyveezymf/MovieOracle.git
cd MovieOracle

# 2. Make sure you have Python 3 with numpy and pandas installed
pip install numpy pandas

# 3. Start the app
bash start.sh

# 4. Open http://localhost:8000 in your browser
```

## Project Structure

```
├── start.sh                  # One-command launcher
├── backend/
│   ├── server.py             # Python HTTP API server
│   ├── model.py              # ML model (Ridge + Logistic regression)
│   ├── data_generator.py     # Synthetic training data generator
│   └── data/
│       ├── movies.csv        # 6,000 film training dataset
│       ├── model.json        # Trained model weights
│       └── talent.json       # Talent database for search
├── frontend/
│   ├── index.html            # Full React UI (served by backend)
│   └── BoxOfficeOracle.jsx   # Standalone React component
└── README.md
```

## How It Works

1. **Data Generation** — Synthetic dataset of 6,000 films with realistic distributions based on real-world box office patterns
2. **Feature Engineering** — Director ROI, producer revenue multipliers, writer quality scores, genre modifiers, budget tiers, release seasonality, and talent interaction features
3. **Model Training** — Ridge regression for revenue prediction + logistic regression for success classification, both implemented in pure NumPy
4. **API** — Python HTTP server exposes `/api/predict`, `/api/talent/*`, and `/api/genres` endpoints
5. **Frontend** — React 18 SPA with Tailwind-inspired styling, searchable dropdowns, SVG gauge, and animated bar charts

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| GET | `/api/talent/directors?q=nolan` | Search directors |
| GET | `/api/talent/producers?q=blum` | Search producers |
| GET | `/api/talent/writers?q=sorkin` | Search writers |
| GET | `/api/genres` | List all genres |
| POST | `/api/predict` | Get financial prediction |

## Sample Predictions

| Scenario | Revenue | ROI | Success Prob |
|----------|---------|-----|-------------|
| Nolan + Thomas, Sci-Fi, $200M | $353M | +76.7% | 52.7% |
| Peele + Blum, Horror, $20M | $46M | +130.3% | 78.7% |
| Cameron + Landau, Sci-Fi, $350M | $791M | +126.1% | 68.8% |
| Unknown + A24, Drama, $5M | $4.2M | -16.2% | 24.6% |

## Tech Stack

- **Backend**: Python 3, NumPy, Pandas (no external ML or web framework dependencies)
- **Frontend**: React 18, vanilla CSS with CSS custom properties
- **ML**: Ridge regression + logistic regression (pure NumPy implementation)

## License

MIT
