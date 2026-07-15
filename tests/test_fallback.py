"""VA-49 — realtime → traditional fallback on failure.

An early realtime failure (nothing delivered yet) re-runs the turn on the traditional
pipeline; the client gets a slow-path answer instead of an error. Mid-stream failures
propagate (replaying would duplicate audio), caller errors propagate, and every fallback is
counted (VA-60).
"""
from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dispatch import Architecture
from app.main import create_app
from app.observability import EventCounters
from app.pipelines.fallback import RealtimeWithFallback
from app.pipelines.factory import build_pipeline_registry
from app.pipelines.realtime import RealtimePipeline
from app.pipelines.traditional import TraditionalPipeline
from app.providers.factory import register_realtime
from app.providers.mock import MockLlm, MockRealtime, MockStt, MockTts
from app.streaming.events import AnswerDelta, AudioChunk, Done, TranscriptFinal
from app.streaming.schemas import VoiceTurnRequest

AUDIO = VoiceTurnRequest.model_validate(
    {"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01\x02").decode()}}
)
TEXT = VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": "hi"}})


class FailingRealtime:
    """Realtime connection that dies before delivering anything."""

    name = "failing"

    async def converse(self, audio_in):
        async for _ in audio_in:
            pass
        raise RuntimeError("realtime connection lost")
        yield  # pragma: no cover — generator marker

    async def interrupt(self) -> None:  # pragma: no cover — protocol parity
        pass


class MidstreamFailingRealtime:
    """Delivers one chunk, then dies — a replay would duplicate played audio."""

    name = "midstream-failing"

    async def converse(self, audio_in):
        async for _ in audio_in:
            pass
        yield b"chunk-0"
        raise RuntimeError("dropped mid-reply")

    async def interrupt(self) -> None:  # pragma: no cover — protocol parity
        pass


def _wrapper(realtime_provider, counters: EventCounters | None = None) -> RealtimeWithFallback:
    traditional = TraditionalPipeline(MockStt(), MockLlm(), MockTts(), counters=counters)
    primary = RealtimePipeline(realtime_provider, counters=counters)
    return RealtimeWithFallback(primary, traditional, counters=counters)


def _drain(stream):
    return asyncio.run(_collect(stream))


async def _collect(stream):
    return [event async for event in stream]


# --- the fallback ------------------------------------------------------------------------------

def test_early_failure_streams_the_traditional_turn():
    counters = EventCounters()
    events = _drain(_wrapper(FailingRealtime(), counters).stream(AUDIO))

    names = [type(e).__name__ for e in events]
    assert "TranscriptFinal" in names  # slow-path events: the STT ran
    assert "AnswerDelta" in names
    assert isinstance(events[-1], Done)
    final = next(e for e in events if isinstance(e, TranscriptFinal))
    assert final.text == "mock transcript"
    answer = "".join(e.text for e in events if isinstance(e, AnswerDelta))
    assert answer == "mock answer"


def test_fallback_is_counted_and_metering_stays_honest():
    counters = EventCounters()
    _drain(_wrapper(FailingRealtime(), counters).stream(AUDIO))
    summary = counters.summary()
    assert summary["realtime"]["turns"] == 1
    assert summary["realtime"]["errors"] == 1  # the primary did fail
    assert summary["realtime"]["fallbacks"] == 1  # and the turn was recovered
    assert summary["realtime"]["fallback_rate"] == 1.0
    assert summary["traditional"]["turns"] == 1  # the recovery ran as a real slow turn


def test_midstream_failure_propagates_without_fallback():
    counters = EventCounters()
    stream = _wrapper(MidstreamFailingRealtime(), counters).stream(AUDIO)

    async def scenario():
        received = []
        with pytest.raises(RuntimeError, match="dropped mid-reply"):
            async for event in stream:
                received.append(event)
        return received

    received = asyncio.run(scenario())
    assert len([e for e in received if isinstance(e, AudioChunk)]) == 1
    assert counters.summary()["realtime"]["fallbacks"] == 0  # no replay after delivery


def test_success_path_untouched():
    counters = EventCounters()
    events = _drain(_wrapper(MockRealtime(), counters).stream(AUDIO))
    assert isinstance(events[0], AudioChunk) and isinstance(events[-1], Done)
    assert counters.summary()["realtime"]["fallbacks"] == 0


def test_caller_errors_do_not_fall_back():
    counters = EventCounters()
    with pytest.raises(ValueError, match="requires audio input"):
        _drain(_wrapper(FailingRealtime(), counters).stream(TEXT))
    assert counters.summary()["realtime"]["fallbacks"] == 0


def test_run_delivery_falls_back_too():
    counters = EventCounters()
    result = asyncio.run(_wrapper(FailingRealtime(), counters).run(AUDIO))
    assert result.answer_text == "mock answer"
    assert counters.summary()["realtime"]["fallbacks"] == 1


# --- factory wiring ----------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    base: dict = dict(
        stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock"
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_factory_wraps_realtime_by_default():
    registry = build_pipeline_registry(_settings())
    assert isinstance(registry.get(Architecture.REALTIME), RealtimeWithFallback)


def test_factory_honours_disable_flag():
    registry = build_pipeline_registry(_settings(realtime_fallback_enabled=False))
    assert isinstance(registry.get(Architecture.REALTIME), RealtimePipeline)


# --- end to end through the app -----------------------------------------------------------------

def test_fast_endpoint_degrades_to_slow_path_over_http():
    register_realtime("failing-e2e", lambda _s: FailingRealtime())
    client = TestClient(create_app(_settings(realtime_provider="failing-e2e")))

    resp = client.post(
        "/api/v1/voice/fast",
        json={"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01").decode()}},
    )
    assert resp.status_code == 200
    # the client got a full slow-path turn over the fast URL
    assert "event: transcript.final" in resp.text
    assert "event: answer.delta" in resp.text
    assert resp.text.rstrip().rindex("event: done") >= 0

    counters = client.get("/api/v1/counters").json()
    assert counters["realtime"]["fallbacks"] == 1
    assert counters["traditional"]["turns"] == 1
