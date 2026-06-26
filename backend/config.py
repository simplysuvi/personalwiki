import os as _os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR     = PROJECT_ROOT / "data"
KB_DIR        = DATA_DIR / "kb"
KB_SOURCES    = KB_DIR / "sources"     # original uploaded files (provenance)
KB_MARKDOWN   = KB_DIR / "markdown"    # converted markdown — the indexed knowledge
DB_DIR        = DATA_DIR / "db"
MODELS_DIR    = PROJECT_ROOT / "models"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in _os.environ:
            _os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return _os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _resolve_path(raw: str) -> Path:
    p = Path(_os.path.expanduser(raw))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _env_path(name: str, default: Path | str) -> Path:
    raw = _os.environ.get(name)
    return _resolve_path(raw) if raw else Path(default)


def _env_optional_path(name: str, default: Path | None) -> Path | None:
    raw = _os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"", "0", "false", "none", "null"}:
        return None
    return _resolve_path(raw)

# ── Identity ─────────────────────────────────────────────────────────────────
USER_NAME      = _env("KB_USER_NAME", "User")
ASSISTANT_NAME = _env("KB_ASSISTANT_NAME", "Assistant")

# ── Reasoning backend ────────────────────────────────────────────────────────
#   "litert"   -> on-device Gemma-4 .litertlm via litert_lm
#   "llamacpp" -> GGUF via llama-cpp-python (Metal)
LLM_BACKEND = _env("LLM_BACKEND", "llamacpp").lower()
LITERT_MODEL_PATH = _env_path("LITERT_MODEL_PATH", Path("~/models/gemma-4-E2B-it.litertlm").expanduser())
LITERT_CONTEXT_TOKENS = _env_int("LITERT_CONTEXT_TOKENS", 4096)
LITERT_OUTPUT_RESERVE = _env_int("LITERT_OUTPUT_RESERVE", 900)
LITERT_TOOL_RESULT_MAX_CHARS = _env_int("LITERT_TOOL_RESULT_MAX_CHARS", 2400)

REASONING_MODEL_PATH = _env_path("REASONING_MODEL_PATH", MODELS_DIR / "gemma-4-12b-it-Q4_K_M.gguf")
EMBEDDING_MODEL_PATH = _env_path("EMBEDDING_MODEL_PATH", MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf")
CHAT_TEMPLATE_PATH   = _env_optional_path("CHAT_TEMPLATE_PATH", TEMPLATES_DIR / "gemma4-tools.jinja")

# ── llama-cpp-python ─────────────────────────────────────────────────────────
N_GPU_LAYERS = _env_int("N_GPU_LAYERS", -1)
N_CTX        = _env_int("N_CTX", 16384)
N_THREADS    = _env_int("N_THREADS", 8)
CONTEXT_RESERVE = _env_int("CONTEXT_RESERVE", 1536)
REASONING_FEEDBACK_CHARS = _env_int("REASONING_FEEDBACK_CHARS", 1500)
KEEP_FULL_TOOL_RESULTS = _env_int("KEEP_FULL_TOOL_RESULTS", 2)

# ── Agent ────────────────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS = _env_int("MAX_AGENT_ITERATIONS", 10)
MAX_FILE_SIZE_CHARS  = _env_int("MAX_FILE_SIZE_CHARS", 8000)

# ── BM25 (search tool) ───────────────────────────────────────────────────────
BM25_TOP_K = _env_int("BM25_TOP_K", 8)
BM25_INDEX_PATH = DB_DIR / "bm25.pkl"
BM25_META_PATH  = DB_DIR / "bm25_meta.pkl"

# ── Converter ────────────────────────────────────────────────────────────────
#   "hybrid"      -> pymupdf4llm for text PDFs, marker-pdf for scanned (OCR)
#   "pymupdf4llm" -> always pymupdf4llm (no OCR)
#   "marker"      -> always marker-pdf (heavy; OCR everything)
PDF_CONVERTER = _env("PDF_CONVERTER", "hybrid").lower()
SCANNED_TEXT_THRESHOLD = _env_int("SCANNED_TEXT_THRESHOLD", 40)   # < this many chars/page on a PDF -> treat as scanned
