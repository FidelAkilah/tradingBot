#!/bin/bash
# ─────────────────────────────────────────────
# Start the trading bot + dashboard locally
# ─────────────────────────────────────────────
# Usage:  ./start.sh
#   - Backend API:  http://localhost:8000
#   - API Docs:     http://localhost:8000/docs
#   - Dashboard:    http://localhost:3000
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and add your Binance API keys:"
    echo "  cp .env.example .env"
    exit 1
fi

# ─── Python Virtual Environment ───
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing Python dependencies (first time only)..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install fastapi "uvicorn[standard]" ccxt numpy aiofiles
    echo "Python setup complete."
    echo ""
fi

# Activate venv for this script
source "$VENV_DIR/bin/activate"

# Quick check — install anything missing
python -c "import fastapi" 2>/dev/null || {
    echo "Installing missing Python dependencies..."
    pip install fastapi "uvicorn[standard]" ccxt numpy aiofiles
}

# ─── Node.js Dashboard ───
if [ ! -d "dashboard/node_modules" ]; then
    echo "Installing dashboard dependencies..."
    (cd dashboard && npm install)
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     CRYPTO SWING TRADING BOT             ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Backend:   http://localhost:8000         ║"
echo "║  API Docs:  http://localhost:8000/docs    ║"
echo "║  Dashboard: http://localhost:3000         ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Press Ctrl+C to stop everything          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

BACKEND_PID=""
FRONTEND_PID=""

# Trap Ctrl+C to kill both processes cleanly
cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    [ -n "$BACKEND_PID" ]  && kill $BACKEND_PID 2>/dev/null
    sleep 2
    [ -n "$FRONTEND_PID" ] && kill -9 $FRONTEND_PID 2>/dev/null
    [ -n "$BACKEND_PID" ]  && kill -9 $BACKEND_PID 2>/dev/null
    wait 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend API server (background) — uses venv python
echo "[1/2] Starting backend API server..."
python server.py &
BACKEND_PID=$!

# Wait for backend to be ready before starting frontend
echo "     Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "     API is ready!"
        break
    fi
    sleep 1
done

# Start frontend dashboard (background)
echo "[2/2] Starting dashboard..."
(cd dashboard && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "  Open http://localhost:3000 in your browser"
echo ""

# Wait for either process to exit
wait
