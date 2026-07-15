"""VA-37 — ground answers strictly in the full-document context (no RAG)."""
import asyncio

from app.context.grounding import (
    GROUNDING_INSTRUCTIONS,
    build_grounded_prompt,
    extract_citations,
    ground_llm,
)
from app.context.loader import DocumentContext
from app.providers.gemini_llm import GeminiLlm


# --- grounding instructions -------------------------------------------------------------

def test_instructions_demand_strict_grounding_citation_and_honest_decline():
    text = GROUNDING_INSTRUCTIONS.lower()
    assert "only" in text                              # answer only from the document
    assert "cite" in text                              # cite the article/clause
    assert "not contained in the document" in text     # decline path
    assert "do not guess" in text


def test_build_grounded_prompt_layers_on_base():
    out = build_grounded_prompt("You are a helpful assistant.")
    assert out.startswith("You are a helpful assistant.")
    assert GROUNDING_INSTRUCTIONS in out


def test_build_grounded_prompt_handles_empty_base():
    assert build_grounded_prompt("") == GROUNDING_INSTRUCTIONS
    assert build_grounded_prompt("   ") == GROUNDING_INSTRUCTIONS


# --- citation extraction (turn trace) ---------------------------------------------------

def test_extract_citations_dedupes_and_normalizes_in_order():
    answer = (
        "Under Article 21 everyone has this right, see also article 19(1)(a); "
        "Article 21 reiterates it."
    )
    assert extract_citations(answer) == ["Article 21", "Article 19(1)(a)"]


def test_extract_citations_handles_articles_plural_and_suffix():
    assert extract_citations("See Articles 14 and Article 21A.") == ["Article 14", "Article 21A"]


def test_extract_citations_none_when_no_reference():
    assert extract_citations("I could not find that in the document.") == []
    assert extract_citations("") == []


# --- wiring onto the LLM ----------------------------------------------------------------

def _collect(agen):
    return asyncio.run(_drain(agen))


async def _drain(agen):
    return [x async for x in agen]


class _FakeStream:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, *, prompt, system, tools, cache_ref=None):
        self.calls.append({"system": system, "cache_ref": cache_ref})
        yield "ok"


async def _fake_cache(_document):
    return "cache-1"


def test_ground_llm_attaches_document_and_instructions():
    stream = _FakeStream()
    llm = GeminiLlm(api_key="k", system_prompt="Base prompt.", stream=stream,
                    create_cache=_fake_cache)
    doc = DocumentContext(text="THE CONSTITUTION TEXT", path="/data/c.pdf",
                          char_count=21, estimated_tokens=6)

    ground_llm(llm, doc)

    assert llm.document_context == "THE CONSTITUTION TEXT"
    assert "Base prompt." in llm.system_prompt
    assert GROUNDING_INSTRUCTIONS in llm.system_prompt


def test_grounded_llm_sends_grounding_prompt_via_cache():
    stream = _FakeStream()
    llm = GeminiLlm(api_key="k", system_prompt="Base prompt.", stream=stream,
                    create_cache=_fake_cache)
    doc = DocumentContext(text="DOCTEXT", path="/data/c.pdf", char_count=7, estimated_tokens=2)
    ground_llm(llm, doc)

    _collect(llm.generate("what does it say?"))

    call = stream.calls[0]
    assert call["cache_ref"] == "cache-1"                 # document served from cache
    assert GROUNDING_INSTRUCTIONS in call["system"]        # grounding instructions applied
    assert "DOCTEXT" not in call["system"]                 # doc not re-inlined (it's cached)
