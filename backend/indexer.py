"""
indexer.py — BM25 keyword index over the KB markdown layer (data/kb/markdown/*.md).

No embeddings, no ChromaDB. One BM25 document per converted file; search returns
the document name + a matching snippet so the agent knows what to read_doc.

CLI:
    python indexer.py        # full rebuild
"""
from __future__ import annotations
import pickle
import re

from rank_bm25 import BM25Okapi

import config


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _doc_name(md_path):
    """data/kb/markdown/foo.pdf.md -> 'foo.pdf' (the KB document name)."""
    rel = md_path.relative_to(config.KB_MARKDOWN).as_posix()
    return rel[:-3] if rel.endswith(".md") else rel


def rebuild_bm25(paths=None) -> int:
    corpus, meta = [], []
    if config.KB_MARKDOWN.exists():
        for p in sorted(config.KB_MARKDOWN.rglob("*.md")):
            text = p.read_text(encoding="utf-8")
            corpus.append(_tokenize(text))
            meta.append({"name": _doc_name(p), "text": text})
    bm25 = BM25Okapi(corpus) if corpus else None
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(config.BM25_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    return len(meta)


def load_bm25():
    try:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(config.BM25_META_PATH, "rb") as f:
            meta = pickle.load(f)
        return bm25, meta
    except FileNotFoundError:
        return None, []


def search(query: str, top_k: int | None = None):
    """Ranked [(name, score, snippet)] over KB documents."""
    top_k = top_k or config.BM25_TOP_K
    bm25, meta = load_bm25()
    if bm25 is None:
        return []
    q = _tokenize(query)
    qset = set(q)
    scores = bm25.get_scores(q)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        low = meta[i]["text"].lower()
        has_term = any(t in low for t in qset)
        # BM25 IDF can be ≤0 for terms common across a tiny corpus — keep a doc
        # whenever it genuinely contains a query term, ranked by score.
        if not has_term and scores[i] <= 0:
            continue
        cand = [l.strip() for l in meta[i]["text"].splitlines()
                if l.strip() and not l.lstrip().startswith("<!--")]
        snippet = next((l for l in cand if any(t in l.lower() for t in qset)),
                       cand[0] if cand else "")[:160]
        out.append((meta[i]["name"], float(scores[i]), snippet))
        if len(out) >= top_k:
            break
    return out


if __name__ == "__main__":
    n = rebuild_bm25()
    _, meta = load_bm25()
    toks = sum(len(_tokenize(m["text"])) for m in meta)
    print(f"BM25 index: {n} documents, {toks} tokens -> {config.BM25_INDEX_PATH}")
