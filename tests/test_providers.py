"""VA-30 — provider adapter interfaces, mock provider, and config-driven factory."""
import asyncio

import pytest

from app.config import Settings
from app.providers import factory
from app.providers.base import LlmProvider, SttProvider, TranscriptChunk, TtsProvider
from app.providers.mock import MockLlm, MockStt, MockTts

S = Settings(_env_file=None)


async def _aiter(items):
    for item in items:
        yield item


async def _collect(agen):
    return [item async for item in agen]


# --- interface conformance --------------------------------------------------------------

def test_mock_providers_satisfy_interfaces():
    assert isinstance(MockStt(), SttProvider)
    assert isinstance(MockLlm(), LlmProvider)
    assert isinstance(MockTts(), TtsProvider)


def test_mock_providers_are_functional():
    async def run():
        stt = await _collect(MockStt().transcribe(_aiter([b"a", b"b"])))
        llm = await _collect(MockLlm().generate("q"))
        tts = await _collect(MockTts().synthesize(_aiter(["x", "y"])))
        return stt, llm, tts

    stt, llm, tts = asyncio.run(run())
    assert [c.text for c in stt] == ["mock transcript", "mock transcript"]
    assert stt[0].is_final is False and stt[-1].is_final is True and stt[-1].is_end_of_turn
    assert "".join(llm) == "mock answer"
    assert tts == [b"x", b"y"]


# --- factory ----------------------------------------------------------------------------

def test_factory_returns_mock_by_name():
    assert isinstance(factory.get_stt("mock", S), MockStt)
    assert isinstance(factory.get_llm("mock", S), MockLlm)
    assert isinstance(factory.get_tts("mock", S), MockTts)


def test_unknown_provider_raises_with_helpful_message():
    with pytest.raises(factory.UnknownProvider) as ei:
        factory.get_stt("does-not-exist", S)
    assert "does-not-exist" in str(ei.value)
    assert "mock" in str(ei.value)  # lists what's available


def test_registration_makes_a_provider_selectable():
    class OtherStt:
        name = "other"

        async def transcribe(self, audio):
            yield TranscriptChunk(text="other", is_final=True)

    factory.register_stt("other-test", lambda _s: OtherStt())
    try:
        assert isinstance(factory.get_stt("other-test", S), OtherStt)
    finally:
        factory._STT.pop("other-test", None)  # keep global registry clean for other tests


# --- config-driven selection (swap by config, not code) ---------------------------------

def test_make_providers_is_config_driven():
    settings = Settings(
        _env_file=None, stt_provider="mock", llm_provider="mock", tts_provider="mock"
    )
    stt, llm, tts = factory.make_providers(settings)
    assert (stt.name, llm.name, tts.name) == ("mock", "mock", "mock")


def test_default_providers_are_not_registered_until_their_tickets():
    # Defaults point at the real providers, which register in VA-31/34/43. Until then the
    # factory fails loudly rather than silently mis-routing.
    settings = Settings(_env_file=None)  # deepgram / gemini / cartesia
    with pytest.raises(factory.UnknownProvider):
        factory.make_providers(settings)
