#!/bin/bash
echo ""
echo "  🎬  BoxOffice Oracle — Starting Up..."
echo "  ──────────────────────────────────────"
echo ""
cd "$(dirname "$0")/backend"
if [ ! -f "data/movies.csv" ]; then
    echo "  📊  Generating training dataset..."
    python3 data_generator.py
    echo ""
fi
if [ ! -f "data/model.json" ]; then
    echo "  🧠  Training prediction model..."
    python3 model.py
    echo ""
fi
PORT=${1:-8000}
echo "  🚀  Starting server on port $PORT..."
echo "  📍  Open http://localhost:$PORT in your browser"
echo ""
echo "  Press Ctrl+C to stop."
echo ""
python3 server.py $PORT
