"""Realtime pipeline: voice-to-voice fast path (VA-48).

Bridges the realtime adapter (VA-46) into the ``Pipeline`` contract: streams mic audio in and
model audio out over one session, emitting ``audio.chunk`` events and propagating ``session_id``
for continuity. Connection teardown is owned by the adapter. Unlike the traditional path there
is no separate transcript/answer — it is voice-to-voice.

The fast endpoint (VA-24) feeds real mic audio over a WebSocket; here the request's audio blob
is the input stream, which keeps the pipeline seam simple and fully mockable.
"""
from __future__ import annotations

import base64
import time
from typing import AsyncIterator, Callable

from app.dispatch import Architecture
from app.pipelines.base import BasePipeline
from app.providers.base import RealtimeProvider
from app.session import SessionStore, TurnState, TurnStateMachine
from app.streaming.events import AnySSEEvent, AudioChunk, Done
from app.streaming.schemas import AudioInput, VoiceTurnRequest, VoiceTurnResult


class RealtimePipeline(BasePipeline):
    """The low-latency voice-to-voice fast path."""

    architecture = Architecture.REALTIME

    def __init__(
        self,
        realtime: RealtimeProvider,
        *,
        session_store: SessionStore | None = None,
        state_factory: Callable[[], TurnStateMachine] = TurnStateMachine,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._realtime = realtime
        # NB: SessionStore defines __len__, so an empty store is falsy — use an explicit
        # None check so an injected (empty) store isn't silently replaced.
        self._sessions = session_store if session_store is not None else SessionStore()
        self._state_factory = state_factory
        self._clock = clock

    async def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        if not isinstance(request.input, AudioInput):
            raise ValueError("the realtime (fast) path requires audio input")

        session = self._sessions.resolve(_tenant_of(request), request.session_id)
        state = self._state_factory()
        started = self._clock()
        state.transition(TurnState.LISTENING)
        state.transition(TurnState.THINKING)

        audio_in = _audio_stream(base64.b64decode(request.input.audio_b64))
        seq = 0
        first_audio: float | None = None
        async for audio in self._realtime.converse(audio_in):
            if first_audio is None:
                first_audio = self._elapsed_ms(started)
                state.transition(TurnState.SPEAKING)
            yield AudioChunk(audio_b64=base64.b64encode(audio).decode("ascii"), seq=seq)
            seq += 1

        state.transition(TurnState.IDLE)
        latency = {"first_audio_ms": first_audio} if first_audio is not None else {}
        yield Done(session_id=session.session_id, latency_ms=latency)

    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        # Realtime is voice-to-voice (no separate transcript/answer text). The four endpoints
        # only use STREAM delivery for the fast path; run() is provided for interface parity.
        latency: dict[str, float] = {}
        session_id = request.session_id
        async for event in self.stream(request):
            if isinstance(event, Done):
                latency = event.latency_ms
                session_id = event.session_id
        return VoiceTurnResult(
            session_id=session_id,
            transcript="",
            answer_text="",
            audio_url=None,
            tools_called=[],
            latency_ms=latency,
        )

    def _elapsed_ms(self, since: float) -> float:
        return round((self._clock() - since) * 1000, 3)


def _tenant_of(request: VoiceTurnRequest) -> str:
    # The auth middleware (VA-15) attaches the validated tenant; default until then.
    return "default"


async def _audio_stream(audio: bytes) -> AsyncIterator[bytes]:
    yield audio
