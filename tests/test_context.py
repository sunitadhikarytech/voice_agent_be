"""VA-35 — full-document context loader (no RAG)."""
import pytest

from app.config import Settings
from app.context import DocumentContext, DocumentError, estimate_tokens, load_document
from app.main import create_app


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


# --- token estimate ---------------------------------------------------------------------

def test_estimate_tokens_is_ceiling_of_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # 5 chars -> ceil(5/4)


# --- loading ----------------------------------------------------------------------------

def test_no_path_means_grounding_off():
    assert load_document(_settings()) is None


def test_loads_and_measures_document(tmp_path):
    doc = tmp_path / "constitution.txt"
    doc.write_text("We the People." * 10, encoding="utf-8")
    result = load_document(_settings(source_doc_path=str(doc)))
    assert isinstance(result, DocumentContext)
    assert result.text == "We the People." * 10
    assert result.char_count == len("We the People.") * 10
    assert result.estimated_tokens == estimate_tokens(result.text)
    assert result.path == str(doc)


def test_missing_configured_document_fails_fast(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(DocumentError) as ei:
        load_document(_settings(source_doc_path=str(missing)))
    assert str(missing) in str(ei.value)


def test_empty_document_fails_fast(tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentError):
        load_document(_settings(source_doc_path=str(doc)))


def test_oversized_document_fails_fast(tmp_path):
    doc = tmp_path / "big.txt"
    doc.write_text("x" * 400, encoding="utf-8")  # ~100 tokens
    with pytest.raises(DocumentError) as ei:
        load_document(_settings(source_doc_path=str(doc), context_window_tokens=10))
    msg = str(ei.value)
    assert "too large" in msg and "10" in msg


# --- startup integration ----------------------------------------------------------------

def test_app_loads_document_at_startup(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("grounding source", encoding="utf-8")
    app = create_app(_settings(source_doc_path=str(doc)))
    assert isinstance(app.state.document, DocumentContext)
    assert app.state.document.text == "grounding source"


def test_app_boots_without_document_by_default():
    app = create_app(_settings())
    assert app.state.document is None


def test_app_fails_to_start_on_missing_document(tmp_path):
    missing = tmp_path / "gone.txt"
    with pytest.raises(DocumentError):
        create_app(_settings(source_doc_path=str(missing)))
