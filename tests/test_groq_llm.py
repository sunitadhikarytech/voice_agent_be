"""VA-34 alt — Groq LLM adapter (mocked stream; no live calls)."""
from __future__ import annotations

import asyncio

from app.config import Settings
from app.context import ground_llm
from app.context.loader import DocumentContext
from app.providers.base import LlmProvider
from app.providers.factory import get_llm
from app.providers.groq_llm import GroqLlm


def _stream_of(*tokens):
    async def _stream(*, messages):
        _stream.messages = messages
        for t in tokens:
            yield t
    return _stream


async def _collect(agen):
    return [x async for x in agen]


# --- streaming ---------------------------------------------------------------------------------

def test_conforms_to_interface():
    assert isinstance(GroqLlm("k"), LlmProvider)


def test_streams_tokens():
    llm = GroqLlm("k", stream=_stream_of("Article ", "14 ", "is ", "equality."))
    out = asyncio.run(_collect(llm.generate("what is article 14?")))
    assert "".join(out) == "Article 14 is equality."


def test_system_prompt_is_sent():
    s = _stream_of("ok")
    llm = GroqLlm("k", system_prompt="You are VANI.", stream=s)
    asyncio.run(_collect(llm.generate("hi")))
    assert s.messages[0] == {"role": "system", "content": "You are VANI."}
    assert s.messages[1] == {"role": "user", "content": "hi"}


def test_per_call_system_override():
    s = _stream_of("ok")
    llm = GroqLlm("k", system_prompt="base", stream=s)
    asyncio.run(_collect(llm.generate("hi", system="OVERRIDE")))
    assert s.messages[0]["content"] == "OVERRIDE"


# --- grounding (inline, capped) ----------------------------------------------------------------

def test_document_is_inlined_into_system():
    s = _stream_of("ok")
    llm = GroqLlm("k", stream=s)
    doc = DocumentContext(text="THE-CONSTITUTION-TEXT", path="/x.pdf", char_count=21, estimated_tokens=6)
    ground_llm(llm, doc)
    asyncio.run(_collect(llm.generate("q")))
    sys_msg = s.messages[0]["content"]
    assert "THE-CONSTITUTION-TEXT" in sys_msg
    assert "SOURCE DOCUMENT" in sys_msg  # grounding framing present


def test_oversized_document_is_truncated_to_budget():
    s = _stream_of("ok")
    big = "x" * 500_000
    llm = GroqLlm("k", max_context_chars=1000, stream=s)
    llm.document_context = big
    asyncio.run(_collect(llm.generate("q")))
    # the inlined doc is capped; the whole system message stays near the budget, not 500k
    assert len(s.messages[0]["content"]) < 2000


def test_no_document_means_plain_system():
    s = _stream_of("ok")
    llm = GroqLlm("k", system_prompt="base", stream=s)
    asyncio.run(_collect(llm.generate("q")))
    assert s.messages[0]["content"] == "base"


# --- wiring ------------------------------------------------------------------------------------

def test_factory_builds_groq_from_config():
    settings = Settings(_env_file=None, llm_provider="groq", groq_api_key="gsk_test")
    llm = get_llm(settings.llm_provider, settings)
    assert isinstance(llm, GroqLlm) and llm.name == "groq"


def test_from_settings_reads_key_and_model():
    settings = Settings(_env_file=None, groq_api_key="gsk_test", groq_model="llama-3.3-70b-versatile")
    llm = GroqLlm.from_settings(settings)
    assert llm._api_key == "gsk_test" and llm._model == "llama-3.3-70b-versatile"
