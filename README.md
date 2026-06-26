# Personal Document KB

A local-first document knowledge base with a React chat UI and a FastAPI agent
backend. Upload PDFs, CSVs, Markdown, text, or JSON files; the app converts them
to Markdown, builds a local BM25 index, and lets a local LLM answer with document
tools. No cloud API is required.

## Project Layout

```text
.
├── backend/              FastAPI app, model loading, conversion, indexing, tools
├── frontend/             React/Vite chat and knowledge-base UI
├── data/
│   ├── kb/               Runtime document state; sources + converted Markdown
│   └── db/               Runtime BM25 index files
├── models/               Local GGUF model weights
├── templates/            Commit-friendly chat templates
├── setup.sh              First-time setup: deps, folders, env, models
├── run.sh                Build frontend and serve the full app
├── requirements.txt      Python dependencies
└── .env.example          Copy to .env for local configuration
```

## What Is Committed

- Source code in `backend/` and `frontend/`
- Setup and run scripts
- The Gemma tool-calling chat template in `templates/`
- Empty `.gitkeep` placeholders for runtime folders

## What Is Not Committed

The following are machine-local runtime state and are ignored by git:

- `models/` model weights and Hugging Face caches
- `data/kb/sources/` uploaded source documents
- `data/kb/markdown/` converted Markdown
- `data/kb/manifest.json`
- `data/db/` generated BM25 indexes
- `.venv/`, `frontend/node_modules/`, `frontend/dist/`

## Requirements

- macOS on Apple Silicon for the default Metal setup
- Python 3.11+
- Node.js 18+
- Enough disk for the selected GGUF model files

## Quick Start

```bash
git clone https://github.com/simplysuvi/personalwiki.git
cd personalwiki
cp .env.example .env
./setup.sh
./run.sh
```

Open `http://localhost:8000` if the browser does not open automatically.

`setup.sh` is safe to rerun. It skips models that are already present. Use
`SKIP_MODEL_DOWNLOAD=1 ./setup.sh` if you want to point `.env` at existing model
files instead of downloading the defaults.

## Configuration

Edit `.env` at the repo root. The common settings are:

- `KB_USER_NAME` and `KB_ASSISTANT_NAME` for UI/prompt labels
- `REASONING_MODEL_PATH` and `EMBEDDING_MODEL_PATH` for local model files
- `REASONING_MODEL_REPO` / `REASONING_MODEL_FILE` and embedding equivalents for setup downloads
- `CHAT_TEMPLATE_PATH` for the tool-calling chat template
- `N_CTX`, `N_THREADS`, and `N_GPU_LAYERS` for llama-cpp runtime tuning
- `PDF_CONVERTER` for `hybrid`, `pymupdf4llm`, or `marker`

Relative paths are resolved from the repository root.

## Running In Development

Backend:

```bash
source .venv/bin/activate
cd backend
python -m uvicorn main:app --port 8000 --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

The Vite dev server proxies API calls to `http://localhost:8000` through the
frontend's `API_BASE` setting.

## Rebuilding The Index

The app rebuilds the BM25 index after uploads and deletions. If you manually
change `data/kb/markdown/`, rebuild it with:

```bash
cd backend
python indexer.py
```
