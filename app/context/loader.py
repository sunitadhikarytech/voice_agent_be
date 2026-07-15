"""Full-document context loader (VA-35, + PDF support).

Loads the entire source document (the "constitution") once at startup and validates it fits
the model's context window. The whole document IS the context — there is no vector store /
RAG. A configured-but-missing (or oversized) document fails fast with a clear error; when no
document is configured, grounding is simply off and the service still boots.

Plain-text (``.txt``/``.md``) and ``.pdf`` sources are supported; a PDF's text is extracted
with ``pypdf``. VA-36 attaches this to the model with prompt caching; VA-37 grounds answers
in it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Rough chars-per-token heuristic for a fast, dependency-free size guard. A real tokenizer is
# not warranted here — this only needs to catch documents that clearly overflow the window.
_CHARS_PER_TOKEN = 4


class DocumentError(RuntimeError):
    """Raised when a configured source document is missing, empty, or too large."""


@dataclass(frozen=True, slots=True)
class DocumentContext:
    """The loaded source document and its measured size."""

    text: str
    path: str
    char_count: int
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (ceiling of chars / chars-per-token)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF with pypdf, concatenating all pages."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a declared dependency
        raise DocumentError("pypdf is required to load PDF source documents") from exc
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise DocumentError(f"could not extract text from PDF {path}: {exc}") from exc


def _read_document(path: Path) -> str:
    """Read the document text, extracting from PDF when the suffix is ``.pdf``."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def load_document(settings) -> DocumentContext | None:
    """Load and validate the source document named by ``settings.source_doc_path``.

    Returns ``None`` when no path is configured (grounding disabled). Raises
    :class:`DocumentError` when a configured path is missing, unreadable, empty, or estimated
    to exceed ``settings.context_window_tokens``.
    """
    raw_path = (settings.source_doc_path or "").strip()
    if not raw_path:
        return None

    path = Path(raw_path)
    if not path.exists():
        raise DocumentError(f"source document not found: {raw_path}")
    try:
        text = _read_document(path)
    except DocumentError:
        raise
    except OSError as exc:
        raise DocumentError(f"could not read source document {raw_path}: {exc}") from exc

    if not text.strip():
        raise DocumentError(f"source document is empty: {raw_path}")

    estimated = estimate_tokens(text)
    window = settings.context_window_tokens
    if estimated > window:
        raise DocumentError(
            f"source document {raw_path} is too large for the context window: "
            f"~{estimated} tokens > {window}"
        )

    return DocumentContext(
        text=text, path=raw_path, char_count=len(text), estimated_tokens=estimated
    )
