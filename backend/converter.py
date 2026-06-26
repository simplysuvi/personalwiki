"""
converter.py — standardized document → markdown conversion.

Every file added to the KB is converted to markdown by type:
  PDF (text layer) -> pymupdf4llm   (fast, offline, clean tables)
  PDF (scanned)    -> marker-pdf    (OCR; lazy-imported, heavy)   [PDF_CONVERTER=hybrid|marker]
  CSV              -> pandas        (markdown table)
  .md/.txt/.json   -> passthrough
The converted markdown is what gets indexed and answered from; the original file
is kept as provenance.
"""
from __future__ import annotations
from pathlib import Path

import config

PDF = {".pdf"}
CSV = {".csv"}
TEXT = {".md", ".markdown", ".txt", ".json"}
SUPPORTED = PDF | CSV | TEXT


def is_supported(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED


def _pdf_chars_per_page(path: Path) -> float:
    import fitz
    d = fitz.open(path)
    try:
        chars = sum(len(pg.get_text()) for pg in d)
        return chars / max(1, d.page_count)
    finally:
        d.close()


def _convert_pdf_pymupdf4llm(path: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path), show_progress=False)


def _convert_pdf_marker(path: Path) -> str:
    """OCR + layout via marker-pdf. Lazy import; models download on first use."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    global _MARKER_MODELS
    try:
        models = _MARKER_MODELS
    except NameError:
        models = _MARKER_MODELS = create_model_dict()
    rendered = PdfConverter(artifact_dict=models)(str(path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _convert_pdf(path: Path) -> tuple[str, str]:
    """Return (markdown, method)."""
    mode = config.PDF_CONVERTER
    scanned = _pdf_chars_per_page(path) < config.SCANNED_TEXT_THRESHOLD
    if mode == "marker" or (mode == "hybrid" and scanned):
        try:
            return _convert_pdf_marker(path), "marker"
        except ImportError:
            note = ("\n\n> ⚠️ This PDF appears scanned (no text layer) and needs OCR. "
                    "Install the OCR converter with `pip install marker-pdf` to extract it.")
            return (_convert_pdf_pymupdf4llm(path) + note), "pymupdf4llm(scanned)"
    return _convert_pdf_pymupdf4llm(path), "pymupdf4llm"


def _convert_csv(path: Path) -> str:
    import pandas as pd
    df = pd.read_csv(path)
    parts = [f"# {path.name}", "",
             f"{len(df)} rows × {len(df.columns)} columns. "
             f"Columns: {', '.join(map(str, df.columns))}", ""]
    if len(df) <= 200:
        parts.append(df.to_markdown(index=False))
    else:
        parts += ["First 100 rows:", df.head(100).to_markdown(index=False), "",
                  "Last 50 rows:", df.tail(50).to_markdown(index=False)]
    return "\n".join(parts)


def convert(path) -> tuple[str, str]:
    """Convert a file to (markdown, method). Raises ValueError if unsupported."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF:
        return _convert_pdf(path)
    if suffix in CSV:
        return _convert_csv(path), "pandas"
    if suffix in TEXT:
        return path.read_text(encoding="utf-8", errors="ignore"), "passthrough"
    raise ValueError(f"Unsupported file type: {suffix}")
