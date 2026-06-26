#!/bin/bash
# First-time local setup. Run from the repository root:
#   ./setup.sh
#
# Optional:
#   SKIP_LLAMA_CPP=1 ./setup.sh       # use an existing llama-cpp-python install
#   SKIP_MODEL_DOWNLOAD=1 ./setup.sh  # install deps only; point .env at models later
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

mkdir -p data/kb/sources data/kb/markdown data/db models

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "[setup] created .env from .env.example"
fi

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if [ "${SKIP_LLAMA_CPP:-0}" != "1" ]; then
  echo "[setup] installing llama-cpp-python with Metal support"
  CMAKE_ARGS="${CMAKE_ARGS:--DGGML_METAL=on}" python -m pip install \
    llama-cpp-python --force-reinstall --no-cache-dir
fi

python -m pip install -r requirements.txt

if [ -d "frontend" ]; then
  (cd frontend && npm install)
fi

download_if_missing() {
  local label="$1"
  local repo="$2"
  local include_file="$3"
  local expected_path="$4"

  if [ -f "$expected_path" ]; then
    echo "[setup] ${label} already exists: ${expected_path}"
    return
  fi

  echo "[setup] downloading ${label}: ${include_file}"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" \
      --include "$include_file" \
      --local-dir models
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" \
      --include "$include_file" \
      --local-dir models
  else
    echo "error: Hugging Face CLI not found after installing requirements." >&2
    echo "       Try: source .venv/bin/activate && python -m pip install -U huggingface_hub" >&2
    return 1
  fi
}

if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  LLM_BACKEND="${LLM_BACKEND:-llamacpp}"
  REASONING_MODEL_REPO="${REASONING_MODEL_REPO:-unsloth/gemma-4-12b-it-GGUF}"
  REASONING_MODEL_FILE="${REASONING_MODEL_FILE:-gemma-4-12b-it-Q4_K_M.gguf}"
  REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-models/$REASONING_MODEL_FILE}"
  EMBEDDING_MODEL_REPO="${EMBEDDING_MODEL_REPO:-nomic-ai/nomic-embed-text-v1.5-GGUF}"
  EMBEDDING_MODEL_FILE="${EMBEDDING_MODEL_FILE:-nomic-embed-text-v1.5.Q4_K_M.gguf}"
  EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-models/$EMBEDDING_MODEL_FILE}"

  if [ "$LLM_BACKEND" = "llamacpp" ]; then
    download_if_missing "reasoning model" "$REASONING_MODEL_REPO" "$REASONING_MODEL_FILE" "$REASONING_MODEL_PATH"
  else
    echo "[setup] skipping GGUF reasoning model download for LLM_BACKEND=${LLM_BACKEND}"
  fi

  download_if_missing "embedding model" "$EMBEDDING_MODEL_REPO" "$EMBEDDING_MODEL_FILE" "$EMBEDDING_MODEL_PATH"
else
  echo "[setup] skipping model download"
fi

echo "[setup] done"
echo "[setup] run: ./run.sh"
