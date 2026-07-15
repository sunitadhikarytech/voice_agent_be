"""VA-48 — realtime pipeline (voice-to-voice fast path)."""
import asyncio
import base64

import pytest

from app.dispatch import Architecture, Delivery, PipelineRegistry, run_turn
from app.pipelines.realtime import RealtimePipeline
from app.providers.mock import MockRealtime
from app.session import SessionStore, TurnState, TurnStateMachine
from app.streaming.events import AudioChunk, Done
from app.streaming.schemas import VoiceTurnRequest


def _audio_request(session_id=None) -> VoiceTurnRequest:
    b64 = base64.b64encode(b"\x01\x02\x03").decode()
    return VoiceTurnRequest.model_validate(
        {"session_id": session_id, "input": {"kind": "audio", "audio_b64": b64}}
    )


def _text_request() -> VoiceTurnRequest:
    return VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": "hi"}})


def _run_stream(pipe, request):
    async def drive():
        return [e async for e in pipe.stream(request)]

    return asyncio.run(drive())


def test_audio_in_audio_out_then_done():
    pipe = RealtimePipeline(MockRealtime())
    events = _run_stream(pipe, _audio_request())
    audio = [e for e in events if isinstance(e, AudioChunk)]
    assert audio and audio[0].audio_b64 == base64.b64encode(b"out:\x01\x02\x03").decode()
    assert audio[0].seq == 0
    assert isinstance(events[-1], Done)


def test_text_input_is_rejected():
    pipe = RealtimePipeline(MockRealtime())
    with pytest.raises(ValueError):
        _run_stream(pipe, _text_request())


def test_session_continuity_across_turns():
    store = SessionStore(id_factory=lambda: "generated")
    pipe = RealtimePipeline(MockRealtime(), session_store=store)
    done1 = _run_stream(pipe, _audio_request(session_id="sess-x"))[-1]
    done2 = _run_stream(pipe, _audio_request(session_id="sess-x"))[-1]
    assert done1.session_id == "sess-x" == done2.session_id
    # a request without a session id gets a fresh generated one
    done3 = _run_stream(pipe, _audio_request())[-1]
    assert done3.session_id == "generated"


def test_drives_state_machine_idle_to_speaking_to_idle():
    machine = TurnStateMachine()
    pipe = RealtimePipeline(MockRealtime(), state_factory=lambda: machine)
    _run_stream(pipe, _audio_request())
    assert [(t.frm, t.to) for t in machine.history] == [
        (TurnState.IDLE, TurnState.LISTENING),
        (TurnState.LISTENING, TurnState.THINKING),
        (TurnState.THINKING, TurnState.SPEAKING),
        (TurnState.SPEAKING, TurnState.IDLE),
    ]


def test_done_reports_first_audio_latency():
    pipe = RealtimePipeline(MockRealtime())
    done = _run_stream(pipe, _audio_request())[-1]
    assert "first_audio_ms" in done.latency_ms and done.latency_ms["first_audio_ms"] >= 0


def test_dispatches_via_run_turn_fast_path():
    pipe = RealtimePipeline(MockRealtime())
    registry = PipelineRegistry()
    registry.register(pipe)

    async def drive():
        stream = await run_turn(
            _audio_request(), architecture=Architecture.REALTIME,
            delivery=Delivery.STREAM, registry=registry,
        )
        return [e async for e in stream]

    events = asyncio.run(drive())
    assert isinstance(events[-1], Done)
