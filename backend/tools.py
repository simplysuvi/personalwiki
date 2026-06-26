"""
tools.py — what the agent can do over the document KB.

The KB is a set of documents, each converted to markdown
(data/kb/markdown/<name>.md). The agent answers from that markdown; originals
live in data/kb/sources/ for provenance.

Tools:
  list_documents  — every doc in the KB (the agent's map)
  read_document   — a doc's full markdown (optional section= / grep= to target it)
  search          — BM25 keyword search across all docs
  query_csv       — filter rows of a CSV document
"""
from __future__ import annotations
import re
from pathlib import Path

import config
import indexer
import kb


# ── helpers ──────────────────────────────────────────────────────────────────
def _resolve(name: str):
    """Find a KB document's markdown by name (exact, then fuzzy)."""
    name = Path(name.strip()).name
    direct = kb.markdown_path(name)
    if direct.exists():
        return name, direct
    # fuzzy: case/space/punctuation-insensitive basename match
    norm = re.sub(r"[^a-z0-9]", "", name.lower())
    norm = norm[:-2] if norm.endswith("md") else norm
    for p in config.KB_MARKDOWN.glob("*.md"):
        doc = p.name[:-3]
        if re.sub(r"[^a-z0-9]", "", doc.lower()) == norm:
            return doc, p
    return None, None


def _truncate(text: str, limit: int = None) -> str:
    limit = limit or config.MAX_FILE_SIZE_CHARS
    return text if len(text) <= limit else text[:limit] + "\n[... truncated — document is longer]"


def _apply_grep(text: str, pattern: str, context: int = 2, max_hits: int = 40) -> str:
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    lines = text.splitlines()
    keep, hits = set(), 0
    for i, line in enumerate(lines):
        if rx.search(line):
            hits += 1
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
            if hits >= max_hits:
                break
    if not keep:
        return f"No lines match '{pattern}' in this document."
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i > prev + 1:
            out.append("  ...")
        out.append(lines[i]); prev = i
    return f"[{hits} matching line(s) for '{pattern}']\n" + "\n".join(out)


def _extract_section(text: str, section: str) -> str:
    lines = text.splitlines()
    start = level = None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m and section.lower() in m.group(2).lower():
            start, level = i, len(m.group(1)); break
    if start is None:
        heads = [l.strip() for l in lines if re.match(r"^#{1,6}\s", l.strip())]
        return f"No section matching '{section}'. Headings:\n" + "\n".join(heads[:30])
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#{1,6})\s", lines[j].strip())
        if m and len(m.group(1)) <= level:
            end = j; break
    return "\n".join(lines[start:end]).strip()


# ── Tool 1: list_documents ───────────────────────────────────────────────────
def list_documents() -> str:
    """List every document in the KB with a one-line preview — the agent's map."""
    files = kb.list_files()
    if not files:
        return "The knowledge base is empty. The user can add documents in the UI."
    lines = [f"{len(files)} document(s) in the KB:"]
    for f in files:
        md = kb.markdown_path(f["name"])
        first = ""
        if md.exists():
            for l in md.read_text(encoding="utf-8").splitlines():
                l = l.strip()
                if l and not l.startswith("<!--"):
                    first = l[:90]; break
        lines.append(f"- {f['name']}  —  {first}")
    return "\n".join(lines)


# ── Tool 2: read_document ────────────────────────────────────────────────────
def read_document(name: str, section: str = "", grep: str = "") -> str:
    """Read a KB document's markdown. For long docs, pass section='heading' or
    grep='pattern' to read only the relevant part."""
    doc, p = _resolve(name)
    if p is None:
        return f"No document named '{name}'. Call list_documents to see what's in the KB."
    text = p.read_text(encoding="utf-8")
    note = "" if doc == Path(name).name else f"[resolved '{name}' -> '{doc}']\n"
    if str(section).strip():
        return note + _extract_section(text, str(section).strip())
    if str(grep).strip():
        return note + _apply_grep(text, str(grep).strip())
    return note + _truncate(text)


# ── Tool 3: search ───────────────────────────────────────────────────────────
def search(query: str) -> str:
    """BM25 keyword search across all KB documents; returns ranked names + snippets."""
    hits = indexer.search(query)
    if not hits:
        return f"Nothing in the KB matches '{query}'. Try list_documents."
    return "\n".join(f"{name}  (score {score:.2f})\n    {snippet}"
                     for name, score, snippet in hits)


# ── Tool 4: query_csv ────────────────────────────────────────────────────────
def query_csv(name: str, contains: str = "", date_from: str = "", date_to: str = "",
              columns: str = "", limit=100) -> str:
    """Filter rows of a CSV document in the KB. contains=substring across text cols;
    date_from/date_to=YYYY-MM-DD on the date column; columns=which to show."""
    import pandas as pd
    src = kb.source_path(Path(name).name)
    if not src.exists() or src.suffix.lower() != ".csv":
        csvs = [f["name"] for f in kb.list_files() if f["name"].lower().endswith(".csv")]
        return f"No CSV named '{name}'. CSV documents: {', '.join(csvs) or 'none'}"
    try:
        df = pd.read_csv(src)
    except Exception as e:
        return f"Could not read CSV: {e}"
    total = len(df)
    if str(date_from).strip() or str(date_to).strip():
        dcol = next((c for c in df.columns if "date" in str(c).lower()), None)
        if dcol is None:
            return f"No date column. Columns: {', '.join(map(str, df.columns))}"
        dates = pd.to_datetime(df[dcol], errors="coerce")
        if str(date_from).strip():
            df = df[dates >= pd.to_datetime(str(date_from))]; dates = dates[dates.index.isin(df.index)]
        if str(date_to).strip():
            df = df[dates <= pd.to_datetime(str(date_to))]
    if str(contains).strip():
        needle = str(contains).strip()
        tcols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object] or list(df.columns)
        mask = None
        for c in tcols:
            mm = df[c].astype(str).str.contains(needle, case=False, na=False, regex=False)
            mask = mm if mask is None else (mask | mm)
        df = df[mask]
    if str(columns).strip():
        want = [c.strip() for c in str(columns).split(",") if c.strip() and c.strip() in df.columns]
        if want:
            df = df[want]
    try:
        lim = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        lim = 200
    n = len(df)
    shown = df.head(lim)
    head = f"[{n} matching rows of {total}{'; first ' + str(len(shown)) if n > len(shown) else ''}]"
    return head + "\n" + shown.to_markdown(index=False)


# ── registry + schemas ───────────────────────────────────────────────────────
TOOLS = {
    "list_documents": list_documents,
    "read_document":  read_document,
    "search":         search,
    "query_csv":      query_csv,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List every document in the knowledge base with a one-line preview. Call this first to see what's available.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_document",
        "description": "Read a KB document's full text (converted markdown). For a long document, pass section='heading name' for one section, or grep='pattern' for matching lines only — saves context.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Document name, e.g. 'lease.pdf'"},
            "section": {"type": "string", "description": "optional: only the section whose heading matches"},
            "grep": {"type": "string", "description": "optional: only lines matching this pattern (+context)"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Keyword search (BM25) across all KB documents. Returns ranked document names + snippets. Use when you don't know which document holds a fact.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords to search for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "query_csv",
        "description": "Filter rows of a CSV document. Example: name='transactions.csv', contains='adp', date_from='2025-01-01', date_to='2025-12-31', columns='date,amount'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "CSV document name"},
            "contains": {"type": "string", "description": "keep rows where any text column contains this"},
            "date_from": {"type": "string", "description": "rows on/after YYYY-MM-DD"},
            "date_to": {"type": "string", "description": "rows on/before YYYY-MM-DD"},
            "columns": {"type": "string", "description": "comma-separated columns to show"},
            "limit": {"type": "integer", "description": "max rows (default 100)"}},
            "required": ["name"]}}},
]
