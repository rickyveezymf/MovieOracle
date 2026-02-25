"""
HTTP API server for Movie Financial Success Predictor.
Uses Python's built-in http.server — no external framework needed.
"""
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from model import MoviePredictor
from data_generator import DIRECTORS, PRODUCERS, WRITERS, GENRE_MODIFIERS, MONTH_MODIFIERS

# ─── XGBoost pipeline detection ──────────────────────────────────────────────
# Attempt to load the trained XGBoost pipeline. Falls back to legacy NumPy
# model automatically if models haven't been trained yet.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

_USE_XGBOOST = False
_xgb_predictor = None
try:
    from models.predict import MoviePredictorXGB
    _models_dir = os.path.join(PROJECT_ROOT, "models")
    _lookup_path = os.path.join(PROJECT_ROOT, "data", "talent_lookup.json")
    if MoviePredictorXGB.is_available(_models_dir):
        _xgb_predictor = MoviePredictorXGB(
            models_dir=_models_dir,
            lookup_path=_lookup_path,
        )
        _USE_XGBOOST = True
        print("  [server] XGBoost pipeline loaded (data-driven predictions).")
    else:
        print("  [server] XGBoost models not found — using legacy model.")
        print("  [server] To upgrade: run data/collect_data.py → data/build_features.py → models/train.py")
except Exception as _e:
    print(f"  [server] XGBoost unavailable ({_e}) — using legacy model.")

# ─── Load legacy model and talent data ───────────────────────────────────────
print("Starting Movie Predictor Server...")
model = MoviePredictor()
model.load("data/model.json")

with open("data/talent.json") as f:
    TALENT_DB = json.load(f)


def _build_search_db_from_lookup(lookup_path: str) -> dict:
    """
    Build a talent search database from the XGBoost talent_lookup.json.
    Returns a dict matching the TALENT_DB format used by the search endpoints.
    Sorted by film_count descending so most prominent talent appears first.
    """
    try:
        with open(lookup_path) as f:
            lookup = json.load(f)

        def _to_list(role_dict, stats_key_map):
            entries = []
            for name, stats in role_dict.items():
                entry = {"name": name, "film_count": stats.get("film_count", 0)}
                for out_key, in_key in stats_key_map.items():
                    val = stats.get(in_key)
                    if val is not None:
                        entry[out_key] = round(val, 2)
                entries.append(entry)
            return sorted(entries, key=lambda x: -x.get("film_count", 0))

        return {
            "directors": _to_list(
                lookup.get("directors", {}),
                {"avg_roi": "avg_roi", "hit_rate": "hit_rate"}
            ),
            "producers": _to_list(
                lookup.get("producers", {}),
                {"hit_rate": "hit_rate"}
            ),
            "writers": _to_list(
                lookup.get("writers", {}),
                {"hit_rate": "hit_rate"}
            ),
            "genres": TALENT_DB.get("genres", [
                "Action", "Adventure", "Animation", "Comedy", "Drama",
                "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
            ]),
        }
    except Exception:
        return TALENT_DB  # Fall back to legacy talent DB


# Use real talent DB if XGBoost pipeline is active
_ACTIVE_TALENT_DB = _build_search_db_from_lookup(_lookup_path) if _USE_XGBOOST else TALENT_DB


def search_talent(talent_list, query, limit=10):
    """Fuzzy search talent by name."""
    q = query.lower().strip()
    if not q:
        return talent_list[:limit]
    return [t for t in talent_list if q in t["name"].lower()][:limit]


def lookup_director(name):
    """Get director features by name."""
    if name in DIRECTORS:
        d = DIRECTORS[name]
        return {"director_avg_roi": d["avg_roi"], "director_hit_rate": d["hit_rate"]}
    return {"director_avg_roi": 0.8, "director_hit_rate": 0.35}


def lookup_producer(name):
    """Get producer features by name."""
    if name in PRODUCERS:
        p = PRODUCERS[name]
        return {"producer_rev_mult": p["avg_rev_mult"], "producer_hit_rate": p["hit_rate"]}
    return {"producer_rev_mult": 1.2, "producer_hit_rate": 0.38}


def lookup_writer(name):
    """Get writer features by name."""
    if name in WRITERS:
        w = WRITERS[name]
        return {"writer_quality": w["quality"], "writer_hit_rate": w["hit_rate"]}
    return {"writer_quality": 0.50, "writer_hit_rate": 0.35}


def get_genre_features(genre):
    """Get genre modifier features."""
    if genre in GENRE_MODIFIERS:
        g = GENRE_MODIFIERS[genre]
        return {"genre_rev_mult": g["rev_mult"], "genre_base_success": g["base_success"]}
    return {"genre_rev_mult": 1.0, "genre_base_success": 0.50}


class APIHandler(SimpleHTTPRequestHandler):
    """Handle API requests and serve frontend."""

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/talent/directors":
            q = params.get("q", [""])[0]
            results = search_talent(_ACTIVE_TALENT_DB["directors"], q)
            self._json_response(results)

        elif path == "/api/talent/producers":
            q = params.get("q", [""])[0]
            results = search_talent(_ACTIVE_TALENT_DB["producers"], q)
            self._json_response(results)

        elif path == "/api/talent/writers":
            q = params.get("q", [""])[0]
            results = search_talent(_ACTIVE_TALENT_DB["writers"], q)
            self._json_response(results)

        elif path == "/api/genres":
            self._json_response(_ACTIVE_TALENT_DB.get("genres", TALENT_DB["genres"]))

        elif path == "/api/health":
            self._json_response({
                "status": "ok",
                "model_loaded": model.revenue_weights is not None,
                "pipeline": "xgboost" if _USE_XGBOOST else "legacy",
                "model_version": "2.0" if _USE_XGBOOST else "1.0",
            })

        elif path == "/" or path == "/index.html":
            self._serve_frontend()

        else:
            # Try serving static files from frontend dir
            frontend_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "frontend",
                path.lstrip("/")
            )
            if os.path.exists(frontend_path):
                self._serve_file(frontend_path)
            else:
                self._serve_frontend()  # SPA fallback

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._json_response({"error": "Invalid JSON"}, 400)
                return

            # Build feature dict from request
            director_name = data.get("director", "Unknown Director")
            producer_name = data.get("producer", "Unknown Producer")
            writer_name = data.get("writer", "Unknown Writer")
            genre = data.get("genre", "Drama")
            budget = data.get("budget", 20_000_000)
            release_month = data.get("release_month", 6)

            if _USE_XGBOOST:
                # ── XGBoost pipeline: data-driven predictions ──
                prediction = _xgb_predictor.predict({
                    "director": director_name,
                    "producer": producer_name,
                    "writer": writer_name,
                    "genre": genre,
                    "budget": budget,
                    "release_month": release_month,
                })
            else:
                # ── Legacy NumPy model: hardcoded heuristics ──
                features = {}
                features.update(lookup_director(director_name))
                features.update(lookup_producer(producer_name))
                features.update(lookup_writer(writer_name))
                features.update(get_genre_features(genre))
                features["budget"] = budget
                features["month_modifier"] = MONTH_MODIFIERS.get(release_month, 1.0)
                prediction = model.predict(features)

            prediction["input"] = {
                "director": director_name,
                "producer": producer_name,
                "writer": writer_name,
                "genre": genre,
                "budget": budget,
                "release_month": release_month,
            }

            self._json_response(prediction)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _serve_frontend(self):
        frontend_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend", "index.html"
        )
        self._serve_file(frontend_path)

    def _serve_file(self, filepath):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            if filepath.endswith(".html"):
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif filepath.endswith(".js"):
                self.send_header("Content-Type", "application/javascript")
            elif filepath.endswith(".css"):
                self.send_header("Content-Type", "text/css")
            elif filepath.endswith(".json"):
                self.send_header("Content-Type", "application/json")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self._set_cors()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._json_response({"error": "File not found"}, 404)

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        if "/api/" in str(args[0]):
            print(f"  API: {args[0]}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = HTTPServer(("0.0.0.0", port), APIHandler)
    print(f"Server running at http://localhost:{port}")
    print(f"API endpoints:")
    print(f"  GET  /api/health")
    print(f"  GET  /api/talent/directors?q=nolan")
    print(f"  GET  /api/talent/producers?q=feige")
    print(f"  GET  /api/talent/writers?q=sorkin")
    print(f"  GET  /api/genres")
    print(f"  POST /api/predict")
    print(f"\nPress Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
