"""
kb.py — the knowledge base: add / remove / list files.

Layout:
  data/kb/sources/<file>        original uploaded document (provenance)
  data/kb/markdown/<file>.md    converted markdown (the indexed knowledge)
  data/kb/manifest.json         per-file metadata (method, size, added time)

Adding a file: save source -> convert to markdown -> reindex.
Removing a file: delete source + markdown -> reindex.
Everything the agent answers from is the markdown layer.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import config
import converter
import indexer

MANIFEST = config.KB_DIR / "manifest.json"


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(m: dict):
    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2))


def _safe_name(filename: str) -> str:
    """Flatten to a basename and strip path separators (no traversal)."""
    return Path(filename).name


def source_path(name: str) -> Path:
    return config.KB_SOURCES / name


def markdown_path(name: str) -> Path:
    return config.KB_MARKDOWN / (name + ".md")


def add_file(filename: str, data: bytes, now: float | None = None) -> dict:
    """Save, convert, index a new document. Returns its manifest record."""
    name = _safe_name(filename)
    if not converter.is_supported(name):
        raise ValueError(f"Unsupported file type: {Path(name).suffix}")
    config.KB_SOURCES.mkdir(parents=True, exist_ok=True)
    config.KB_MARKDOWN.mkdir(parents=True, exist_ok=True)

    src = source_path(name)
    src.write_bytes(data)

    md_text, method = converter.convert(src)
    md_text = md_text or ""
    header = f"<!-- source: {name} -->\n\n"
    markdown_path(name).write_text(header + md_text, encoding="utf-8")

    rec = {
        "name": name,
        "bytes": len(data),
        "method": method,
        "chars": len(md_text),
        "indexed": bool(md_text.strip()),
        "added": now if now is not None else time.time(),
    }
    m = _load_manifest()
    m[name] = rec
    _save_manifest(m)
    indexer.rebuild_bm25()
    return rec


def remove_file(name: str) -> bool:
    """Delete a document's source + markdown and reindex. True if it existed."""
    name = _safe_name(name)
    existed = False
    for p in (source_path(name), markdown_path(name)):
        if p.exists():
            p.unlink()
            existed = True
    m = _load_manifest()
    if name in m:
        del m[name]
        _save_manifest(m)
        existed = True
    if existed:
        indexer.rebuild_bm25()
    return existed


def list_files() -> list[dict]:
    """All KB documents with status, newest first."""
    m = _load_manifest()
    out = []
    for src in sorted(config.KB_SOURCES.glob("*")):
        if not src.is_file() or src.name.startswith("."):
            continue
        rec = m.get(src.name, {})
        out.append({
            "name": src.name,
            "bytes": rec.get("bytes", src.stat().st_size),
            "method": rec.get("method", "?"),
            "chars": rec.get("chars", 0),
            "indexed": rec.get("indexed", markdown_path(src.name).exists()),
            "added": rec.get("added", src.stat().st_mtime),
        })
    out.sort(key=lambda r: r["added"], reverse=True)
    return out


def count() -> int:
    return sum(1 for p in config.KB_SOURCES.glob("*") if p.is_file() and not p.name.startswith("."))
