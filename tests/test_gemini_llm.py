"""VA-34 / VA-36 — Gemini LLM adapter (injected token stream + cache; no live calls)."""
import asyncio

from app.config import Settings
from app.providers.base import LlmProvider
from app.providers.gemini_llm import GeminiLlm


class FakeStream:
    """Records how it was called (incl. cache_ref) and yields canned tokens."""

    def __init__(self, tokens=("Hello", ", ", "world")):
        self._tokens = tokens
        self.calls: list[dict] = []

    async def __call__(self, *, prompt, system, tools, cache_ref=None):
        self.calls.append(
            {"prompt": prompt, "system": system, "tools": tools, "cache_ref": cache_ref}
        )
        for token in self._tokens:
            yield token


class FakeCache:
    """Records the documents it was asked to cache; returns a fixed handle."""

    def __init__(self, handle="cache-1"):
        self._handle = handle
        self.docs: list[str] = []

    async def __call__(self, document: str) -> str:
        self.docs.append(document)
        return self._handle


async def _collect(agen):
    return [item async for item in agen]


def _llm(**kwargs):
    stream = FakeStream(kwargs.pop("tokens", ("Hello", ", ", "world")))
    cache = FakeCache()
    llm = GeminiLlm(api_key="k", stream=stream, create_cache=cache, **kwargs)
    return llm, stream, cache


def test_conforms_to_interface():
    llm, _, _ = _llm()
    assert isinstance(llm, LlmProvider)


def test_generate_streams_tokens():
    llm, _, cache = _llm()
    tokens = asyncio.run(_collect(llm.generate("hi")))
    assert "".join(tokens) == "Hello, world"
    assert cache.docs == []  # no document -> no cache


def test_default_system_prompt_used():
    llm, stream, _ = _llm(system_prompt="SYS-DEFAULT")
    asyncio.run(_collect(llm.generate("hi")))
    assert stream.calls[0]["system"] == "SYS-DEFAULT"
    assert stream.calls[0]["cache_ref"] is None


def test_system_prompt_override_changes_behaviour():
    llm, stream, _ = _llm(system_prompt="SYS-DEFAULT")
    asyncio.run(_collect(llm.generate("hi", system="SYS-OVERRIDE")))
    assert stream.calls[0]["system"] == "SYS-OVERRIDE"


def test_tools_hook_is_passed_through():
    llm, stream, _ = _llm()
    llm.tools = [{"name": "book_appointment"}]
    asyncio.run(_collect(llm.generate("hi")))
    assert stream.calls[0]["tools"] == [{"name": "book_appointment"}]


# --- VA-36: prompt caching of the full document ------------------------------------------

def test_document_is_cached_once_and_reused_across_turns():
    llm, stream, cache = _llm(system_prompt="SYS")
    llm.document_context = "THE-CONSTITUTION-TEXT"
    asyncio.run(_collect(llm.generate("q1")))
    asyncio.run(_collect(llm.generate("q2")))

    # cached exactly once, from the document
    assert cache.docs == ["THE-CONSTITUTION-TEXT"]
    # both turns referenced the same cache handle
    assert [c["cache_ref"] for c in stream.calls] == ["cache-1", "cache-1"]
    # the large document is NOT re-sent in the per-turn system prompt (that's the whole point)
    assert all("THE-CONSTITUTION-TEXT" not in c["system"] for c in stream.calls)


def test_caching_disabled_inlines_the_document():
    llm, stream, cache = _llm(system_prompt="SYS", enable_caching=False)
    llm.document_context = "DOC-TEXT"
    asyncio.run(_collect(llm.generate("q")))

    assert cache.docs == []  # never cached
    assert stream.calls[0]["cache_ref"] is None
    assert "DOC-TEXT" in stream.calls[0]["system"]  # inlined fallback


def test_per_call_system_override_bypasses_cache():
    llm, stream, cache = _llm(system_prompt="SYS")
    llm.document_context = "DOC-TEXT"
    asyncio.run(_collect(llm.generate("q", system="OVERRIDE")))

    assert cache.docs == []  # override can't reuse a fixed-system cache -> inline
    assert stream.calls[0]["cache_ref"] is None
    assert "OVERRIDE" in stream.calls[0]["system"]
    assert "DOC-TEXT" in stream.calls[0]["system"]


def test_from_settings_reads_config():
    settings = Settings(
        _env_file=None,
        gemini_model="gemini-2.0-flash",
        gemini_system_prompt="from-config",
        gemini_enable_prompt_caching=False,
    )
    llm = GeminiLlm.from_settings(settings)
    assert llm.name == "gemini"
    assert llm._model == "gemini-2.0-flash"
    assert llm.system_prompt == "from-config"
    assert llm._enable_caching is False
