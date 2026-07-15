"""VA-47 — realtime barge-in / interruption handling (server half).

A new turn on a session whose previous reply is still *speaking* cancels the in-flight
model response (``RealtimeProvider.interrupt``) and drives the previous turn's state
machine through ``barge_in()``. Sequential turns, other sessions, and turns that haven't
produced audio yet are never interrupted.
"""
from __future__ import annotations

import asyncio
import base64

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipelines.realtime import RealtimePipeline
from app.session import SessionStore, TurnState, TurnStateMachine
from app.streaming.events import AudioChunk, Done
from app.streaming.schemas import VoiceTurnRequest

AUDIO_REQ = {"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01\x02").decode()}}


def _request(session_id: str = "s1") -> VoiceTurnRequest:
    return VoiceTurnRequest.model_validate({"session_id": session_id, **AUDIO_REQ})


class ControlledRealtime:
    """Yields two chunks, then speaks until interrupted — like a long model reply."""

    name = "controlled"

    def __init__(self) -> None:
        self.interrupts = 0
        self._cancelled = asyncio.Event()

    async def converse(self, audio_in):
        async for _ in audio_in:
            pass
        yield b"reply-chunk-0"
        yield b"reply-chunk-1"
        await self._cancelled.wait()  # long reply: keeps "speaking" until cancelled

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._cancelled.set()


class SilentRealtime:
    """Never yields audio until released — models a turn still thinking."""

    name = "silent"

    def __init__(self) -> None:
        self.interrupts = 0
        self.released = asyncio.Event()

    async def converse(self, audio_in):
        async for _ in audio_in:
            pass
        await self.released.wait()
        yield b"late-reply"

    async def interrupt(self) -> None:
        self.interrupts += 1


def _tracking_factory(created: list[TurnStateMachine]):
    def factory() -> TurnStateMachine:
        machine = TurnStateMachine()
        created.append(machine)
        return machine

    return factory


# --- the barge-in itself -----------------------------------------------------------------------

def test_new_turn_interrupts_a_speaking_session():
    async def scenario():
        provider = ControlledRealtime()
        machines: list[TurnStateMachine] = []
        pipeline = RealtimePipeline(
            provider, session_store=SessionStore(), state_factory=_tracking_factory(machines)
        )

        first = pipeline.stream(_request("s1"))
        chunk = await first.__anext__()  # first audio arrives → the turn is SPEAKING
        assert isinstance(chunk, AudioChunk)
        assert machines[0].state is TurnState.SPEAKING

        second = pipeline.stream(_request("s1"))  # the user talks over the reply
        await second.__anext__()  # starting the new turn triggers the barge-in

        assert provider.interrupts == 1
        # speaking → interrupted → listening recorded on the interrupted turn
        transitions = [(t.frm, t.to) for t in machines[0].history]
        assert (TurnState.SPEAKING, TurnState.INTERRUPTED) in transitions
        assert machines[0].state is TurnState.LISTENING

        # both streams still terminate cleanly with done
        first_rest = [e async for e in first]
        second_rest = [e async for e in second]
        assert isinstance(first_rest[-1], Done)
        assert isinstance(second_rest[-1], Done)

    asyncio.run(scenario())


def test_sequential_turns_are_not_interrupted():
    async def scenario():
        provider = ControlledRealtime()
        provider._cancelled.set()  # replies end on their own → turns complete sequentially
        pipeline = RealtimePipeline(provider, session_store=SessionStore())
        for _ in range(2):
            events = [e async for e in pipeline.stream(_request("s1"))]
            assert isinstance(events[-1], Done)
        assert provider.interrupts == 0

    asyncio.run(scenario())


def test_other_sessions_are_never_interrupted():
    async def scenario():
        provider = ControlledRealtime()
        pipeline = RealtimePipeline(provider, session_store=SessionStore())

        first = pipeline.stream(_request("session-a"))
        await first.__anext__()  # session-a is speaking

        second = pipeline.stream(_request("session-b"))  # a different conversation
        await second.__anext__()

        assert provider.interrupts == 0  # barge-in is per session, not global

    asyncio.run(scenario())


def test_a_turn_that_is_not_yet_speaking_is_not_interrupted():
    async def scenario():
        provider = SilentRealtime()
        pipeline = RealtimePipeline(provider, session_store=SessionStore())

        events: list = []

        async def run_first():
            async for event in pipeline.stream(_request("s1")):
                events.append(event)

        task = asyncio.create_task(run_first())
        await asyncio.sleep(0.01)  # let the first turn reach THINKING (no audio yet)

        second = pipeline.stream(_request("s1"))
        provider.released.set()  # unblock replies
        [e async for e in second]
        await task

        assert provider.interrupts == 0  # nothing was playing — no barge-in
        assert isinstance(events[-1], Done)

    asyncio.run(scenario())


def test_finished_turns_leave_no_active_state():
    async def scenario():
        provider = ControlledRealtime()
        provider._cancelled.set()
        pipeline = RealtimePipeline(provider, session_store=SessionStore())
        [e async for e in pipeline.stream(_request("s1"))]
        assert pipeline._active_turns == {}  # deregistered at turn end

    asyncio.run(scenario())


# --- through the app ---------------------------------------------------------------------------

def test_sequential_fast_turns_via_http_do_not_false_positive():
    client = TestClient(
        create_app(
            Settings(
                _env_file=None,
                stt_provider="mock", llm_provider="mock",
                tts_provider="mock", realtime_provider="mock",
            )
        )
    )
    body = {"session_id": "http-s1", **AUDIO_REQ}
    for _ in range(2):
        resp = client.post("/api/v1/voice/fast", json=body)
        assert resp.status_code == 200
        assert "event: done" in resp.text
    # MockRealtime counts interrupts; sequential turns must not trigger any
    from app.dispatch import Architecture

    pipeline = client.app.state.pipelines.get(Architecture.REALTIME)
    pipeline = getattr(pipeline, "_primary", pipeline)  # unwrap the VA-49 fallback wrapper
    assert pipeline._realtime.interrupts == 0
