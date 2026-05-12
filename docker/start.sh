#!/usr/bin/env bash
set -euo pipefail

cd /app
mkdir -p out/api out/demo

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-7860}"

echo "[START] FastAPI on ${API_HOST}:${API_PORT}"
uvicorn src.agent.app:app --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!

cleanup() {
  echo "[STOP] Shutting down processes..."
  if kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

echo "[START] Gradio on ${UI_HOST}:${UI_PORT}"
python gradio_ui.py --host "${UI_HOST}" --port "${UI_PORT}"
