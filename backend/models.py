"""
models.py — load the GGUF models in-process via llama-cpp-python, once.

Both models are process-level singletons created at FastAPI startup (lifespan),
never per-request. If a model file is missing, raise a descriptive error with
the exact download command.
"""
from __future__ import annotations

import config

_INSTALL_HELP = (
    "llama-cpp-python is not installed. On Apple Silicon install with Metal:\n"
    '  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python '
    "--force-reinstall --no-cache-dir"
)


def _llama_cpp():
    """Import llama_cpp lazily so the module loads without it (error surfaces at model load)."""
    try:
        import llama_cpp
        return llama_cpp
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HELP) from e


_llm = None
_embedder = None

def _download_help(kind: str) -> str:
    if kind == "reasoning":
        return (
            "hf download unsloth/gemma-4-12b-it-GGUF \\\n"
            f'  --include "{config.REASONING_MODEL_PATH.name}" --local-dir {config.MODELS_DIR}'
        )
    return (
        "hf download nomic-ai/nomic-embed-text-v1.5-GGUF \\\n"
        f'  --include "{config.EMBEDDING_MODEL_PATH.name}" --local-dir {config.MODELS_DIR}'
    )


def _require(path, kind: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{kind.capitalize()} model not found at:\n  {path}\n"
            f"Download it with:\n  {_download_help(kind)}"
        )


def metal_available() -> bool:
    """True if this llama-cpp build can offload layers to the GPU (Metal)."""
    try:
        return bool(_llama_cpp().llama_supports_gpu_offload())
    except Exception:
        return False


def get_llm():
    global _llm
    if _llm is None:
        _require(config.REASONING_MODEL_PATH, "reasoning")
        if not metal_available():
            print("[models] WARNING: this llama-cpp build has NO GPU offload — "
                  "reinstall with CMAKE_ARGS=\"-DGGML_METAL=on\" for Metal speed.")
        _llm = _llama_cpp().Llama(
            model_path=str(config.REASONING_MODEL_PATH),
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            verbose=False,
        )
        _maybe_attach_chat_template(_llm)
        print(f"[models] reasoning model loaded: {config.REASONING_MODEL_PATH.name} "
              f"(n_ctx={config.N_CTX}, metal={metal_available()})")
    return _llm


def using_custom_template() -> bool:
    p = getattr(config, "CHAT_TEMPLATE_PATH", None)
    return bool(p) and p.exists()


def _maybe_attach_chat_template(llm):
    """Attach a custom Jinja chat template (config.CHAT_TEMPLATE_PATH) so message
    formatting matches the model's trained tool-calling DSL. bos/eos are read
    from the loaded model's own vocab, so this works for any GGUF."""
    if not using_custom_template():
        return
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        template = config.CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
        eos = llm.detokenize([llm.token_eos()], special=True).decode("utf-8", "ignore")
        bos = llm.detokenize([llm.token_bos()], special=True).decode("utf-8", "ignore")
        formatter = Jinja2ChatFormatter(
            template=template, eos_token=eos or "<turn|>", bos_token=bos or "",
            stop_token_ids=[llm.token_eos()],
        )
        llm.chat_handler = formatter.to_chat_handler()
        print(f"[models] custom chat template attached: {config.CHAT_TEMPLATE_PATH.name} "
              f"(eos={eos!r})")
    except Exception as e:
        print(f"[models] WARNING: could not attach custom chat template: {e}")


def get_embedder():
    global _embedder
    if _embedder is None:
        _require(config.EMBEDDING_MODEL_PATH, "embedding")
        _embedder = _llama_cpp().Llama(
            model_path=str(config.EMBEDDING_MODEL_PATH),
            embedding=True,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=2048,
            verbose=False,
        )
        print(f"[models] embedding model loaded: {config.EMBEDDING_MODEL_PATH.name}")
    return _embedder


# ── LiteRT-LM engine (Gemma-4 .litertlm) ─────────────────────────────────────
_litert_engine = None
_litert_cm = None


def get_litert_engine():
    """Singleton LiteRT-LM engine for the 'litert' backend."""
    global _litert_engine, _litert_cm
    if _litert_engine is None:
        import os
        import litert_lm
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        if not os.path.exists(config.LITERT_MODEL_PATH):
            raise FileNotFoundError(
                f"LiteRT model not found at {config.LITERT_MODEL_PATH}"
            )
        _litert_cm = litert_lm.Engine(config.LITERT_MODEL_PATH)
        _litert_engine = _litert_cm.__enter__()
        print(f"[models] LiteRT engine loaded: {config.LITERT_MODEL_PATH}")
    return _litert_engine


def reasoning_model_name() -> str:
    if config.LLM_BACKEND == "litert":
        import os
        return os.path.basename(config.LITERT_MODEL_PATH)
    return config.REASONING_MODEL_PATH.name


def load_reasoning():
    """Load whichever reasoning backend config selects."""
    if config.LLM_BACKEND == "litert":
        return get_litert_engine()
    return get_llm()


def complete(prompt: str, max_tokens: int = 8192, temperature: float = 0.2) -> str:
    """One-shot non-streaming completion on the configured reasoning backend."""
    if config.LLM_BACKEND == "litert":
        engine = get_litert_engine()
        with engine.create_conversation() as conv:
            resp = conv.send_message(prompt)
            return "".join(i.get("text", "") for i in resp.get("content", [])
                           if i.get("type") == "text")
    llm = get_llm()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return out["choices"][0]["message"].get("content") or ""


def models_loaded() -> bool:
    reasoning = (_litert_engine is not None) if config.LLM_BACKEND == "litert" else (_llm is not None)
    return reasoning and _embedder is not None


if __name__ == "__main__":
    print("metal_available:", metal_available())
    llm = get_llm()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Say 'ready' and nothing else."}],
        max_tokens=8,
    )
    print("test inference:", out["choices"][0]["message"]["content"].strip())
    get_embedder()
    emb = _embedder.create_embedding("hello")
    print("embedding dims:", len(emb["data"][0]["embedding"]))
