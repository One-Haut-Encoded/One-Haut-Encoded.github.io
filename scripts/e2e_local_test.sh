#!/usr/bin/env bash
# End-to-end check: static frontend (port 8080) + FastAPI backend (7860).
# Run from repository root: bash scripts/e2e_local_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

UVICORN="${ROOT}/.venv/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  UVICORN="uvicorn"
fi
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

echo "[e2e] Starting backend on 127.0.0.1:7860 ..."
"$UVICORN" backend.main:app --host 127.0.0.1 --port 7860 &
UV_PID=$!

echo "[e2e] Starting static server on 127.0.0.1:8080 ..."
"$PY" -m http.server 8080 --bind 127.0.0.1 --directory "$ROOT" &
HTTP_PID=$!

cleanup() {
  kill "$UV_PID" "$HTTP_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:7860/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

ORIGIN="http://127.0.0.1:8080"
echo "[e2e] GET /health"
curl -sf -H "Origin: $ORIGIN" "http://127.0.0.1:7860/health" | head -c 200
echo

echo "[e2e] OPTIONS /recommend (CORS preflight)"
curl -sS -D - -o /dev/null -X OPTIONS \
  -H "Origin: $ORIGIN" \
  -H "Access-Control-Request-Method: GET" \
  "http://127.0.0.1:7860/recommend?customer_id=x&model=knn&k=12" | head -20

CID="4308983955108b3af43ec57f0557211e44462a5633238351fff14c8b51f16093"
echo "[e2e] GET /recommend (same as app.js fetch)"
curl -sf -H "Origin: $ORIGIN" \
  "http://127.0.0.1:7860/recommend?customer_id=${CID}&model=knn&k=12" | head -c 400
echo

echo "[e2e] GET /recommend_from_selection"
curl -sf -H "Origin: $ORIGIN" \
  "http://127.0.0.1:7860/recommend_from_selection?article_ids=0832253001,0851317001,0717464001&k=12" | head -c 400
echo

echo "[e2e] Fetch index.html from static server (frontend entry)"
curl -sf "http://127.0.0.1:8080/index.html" | head -c 120
echo "..."

echo "[e2e] OK — open http://127.0.0.1:8080 in a browser with backend running to test UI."
