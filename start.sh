#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Start the trading bot + dashboard locally
# ─────────────────────────────────────────────
# Usage:  ./start.sh
#   - Backend API:  http://localhost:8000
#   - API Docs:     http://localhost:8000/docs
#   - Dashboard:    http://localhost:3000
# ─────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: start.sh must be run with bash."
    echo "On Windows PowerShell, use: ./start.ps1 (or ./start.cmd)"
    exit 1
fi

# Detect venv paths for POSIX vs Git Bash on Windows.
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
case "$UNAME_S" in
    MINGW*|MSYS*|CYGWIN*)
        VENV_PYTHON=".venv/Scripts/python.exe"
        VENV_PIP=".venv/Scripts/pip.exe"
        VENV_ACTIVATE=".venv/Scripts/activate"
        PYTHON_BOOTSTRAP="python"
        ;;
    *)
        VENV_PYTHON=".venv/bin/python"
        VENV_PIP=".venv/bin/pip"
        VENV_ACTIVATE=".venv/bin/activate"
        PYTHON_BOOTSTRAP="python3"
        ;;
esac

# Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and add your Binance API keys:"
    echo "  cp .env.example .env"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm is not installed or not in PATH."
    echo "Install Node.js, then run this script again."
    exit 1
fi

# ─── Python Virtual Environment ───
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
        echo "ERROR: $PYTHON_BOOTSTRAP is not installed or not in PATH."
        echo "Install Python, then run this script again."
        exit 1
    fi

    "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
    echo "Installing Python dependencies (first time only)..."
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -r requirements.txt
    echo "Python setup complete."
    echo ""
fi

# Activate venv for this script
source "$VENV_ACTIVATE"

# Quick check — install anything missing
python -c "import fastapi" 2>/dev/null || {
    echo "Installing missing Python dependencies..."
    pip install -r requirements.txt
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
    if command -v curl >/dev/null 2>&1; then
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "     API is ready!"
            break
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -qO- http://localhost:8000/health > /dev/null 2>&1; then
            echo "     API is ready!"
            break
        fi
    else
        # If no HTTP client is available, don't block startup forever.
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
