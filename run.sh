#!/bin/bash
# One-command launcher: builds the UI if needed, starts the server, opens the browser.
# Usage: ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. venv
if [ ! -x ".venv/bin/python" ]; then
  echo "error: .venv not found — create it first (see README)" >&2
  exit 1
fi
source .venv/bin/activate

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8787}"

# 1a. validate local runtime files before doing slower work
python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "backend")
import config

missing = []

if config.LLM_BACKEND == "litert":
    if not Path(config.LITERT_MODEL_PATH).exists():
        missing.append(("LiteRT reasoning model", Path(config.LITERT_MODEL_PATH)))
else:
    if not config.REASONING_MODEL_PATH.exists():
        missing.append(("reasoning model", config.REASONING_MODEL_PATH))

if not config.EMBEDDING_MODEL_PATH.exists():
    missing.append(("embedding model", config.EMBEDDING_MODEL_PATH))

if config.CHAT_TEMPLATE_PATH and not config.CHAT_TEMPLATE_PATH.exists():
    missing.append(("chat template", config.CHAT_TEMPLATE_PATH))

if missing:
    print("error: required local runtime files are missing", file=sys.stderr)
    for label, path in missing:
        print(f"  - {label}: {path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Run ./setup.sh to install dependencies and download the default models.", file=sys.stderr)
    print("You can also edit .env to point at existing model files.", file=sys.stderr)
    raise SystemExit(2)
PY

# 1b. fail fast if the port is already taken (don't waste a 30s model load)
EXISTING=$(lsof -ti tcp:"$APP_PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "error: port $APP_PORT is already in use by PID(s): $EXISTING" >&2
  echo "       a server may already be running. Free it with:" >&2
  echo "         kill $EXISTING" >&2
  echo "       (or open http://localhost:$APP_PORT if that's the server you want)." >&2
  echo "       To use another port, set APP_PORT in .env." >&2
  exit 1
fi

# 2. build the UI if dist/ is missing or any source file is newer than the build
NEED_BUILD=0
if [ ! -f "frontend/dist/index.html" ]; then
  NEED_BUILD=1
else
  NEWER=$(find frontend -newer frontend/dist/index.html \
            -not -path "frontend/dist/*" -not -path "frontend/node_modules/*" \
            \( -name "*.jsx" -o -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" \) \
            -print -quit 2>/dev/null)
  [ -n "$NEWER" ] && NEED_BUILD=1
fi
if [ "$NEED_BUILD" = "1" ]; then
  echo "[run] building UI..."
  [ -d "frontend/node_modules" ] || (cd frontend && npm install --no-audit --no-fund)
  (cd frontend && npm run build)
fi

# 3. open the browser once the server is up (models take ~30s to load)
(
  for _ in $(seq 1 120); do
    if curl -s --max-time 1 "http://localhost:$APP_PORT/api/status" >/dev/null 2>&1; then
      open "http://localhost:$APP_PORT"
      exit 0
    fi
    sleep 1
  done
) &

# 4. run the server in the foreground (Ctrl+C stops everything)
echo "[run] launching - UI will open at http://localhost:$APP_PORT when models finish loading"
cd backend
exec python -m uvicorn main:app --host "$APP_HOST" --port "$APP_PORT"
