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
import logging
import time
from typing import AsyncIterator, Callable

from app.auth import current_tenant
from app.dispatch import Architecture
from app.observability import (
    EventCounters,
    LatencyMetrics,
    UsageMetrics,
    audio_seconds,
    bind_log_context,
)
from app.pipelines.base import BasePipeline
from app.providers.base import RealtimeProvider
from app.session import SessionStore, TurnState, TurnStateMachine
from app.streaming.events import AnySSEEvent, AudioChunk, Done
from app.streaming.schemas import AudioInput, VoiceTurnRequest, VoiceTurnResult

logger = logging.getLogger("app.pipelines.realtime")


class RealtimePipeline(BasePipeline):
    """The low-latency voice-to-voice fast path."""

    architecture = Architecture.REALTIME

    def __init__(
        self,
        realtime: RealtimeProvider,
        *,
        session_store: SessionStore | None = None,
        metrics: LatencyMetrics | None = None,
        usage: UsageMetrics | None = None,
        counters: EventCounters | None = None,
        state_factory: Callable[[], TurnStateMachine] = TurnStateMachine,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._realtime = realtime
        # NB: SessionStore defines __len__, so an empty store is falsy — use an explicit
        # None check so an injected (empty) store isn't silently replaced.
        self._sessions = session_store if session_store is not None else SessionStore()
        self._metrics = metrics
        self._usage = usage
        self._counters = counters
        self._state_factory = state_factory
        self._clock = clock

    async def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        """Count the turn (and any error) around the streaming implementation (VA-60)."""
        if self._counters is not None:
            self._counters.turn(self.architecture.value)
        try:
            async for event in self._stream_impl(request):
                yield event
        except Exception:
            if self._counters is not None:
                self._counters.error(self.architecture.value)
            raise

    async def _stream_impl(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        if not isinstance(request.input, AudioInput):
            raise ValueError("the realtime (fast) path requires audio input")

        tenant = _tenant_of(request)
        session = self._sessions.resolve(tenant, request.session_id)
        bind_log_context(session_id=session.session_id, tenant_id=tenant)
        state = self._state_factory()
        started = self._clock()
        state.transition(TurnState.LISTENING)
        state.transition(TurnState.THINKING)

        input_bytes = base64.b64decode(request.input.audio_b64)
        audio_in = _audio_stream(input_bytes)
        seq = 0
        first_audio: float | None = None
        output_audio_bytes = 0
        async for audio in self._realtime.converse(audio_in):
            if first_audio is None:
                first_audio = self._elapsed_ms(started)
                state.transition(TurnState.SPEAKING)
            output_audio_bytes += len(audio)
            yield AudioChunk(audio_b64=base64.b64encode(audio).decode("ascii"), seq=seq)
            seq += 1

        state.transition(TurnState.IDLE)
        latency = {"first_audio_ms": first_audio} if first_audio is not None else {}
        if self._metrics is not None:
            self._metrics.record(self.architecture.value, latency)
        if self._usage is not None:
            seconds = audio_seconds(len(input_bytes)) + audio_seconds(output_audio_bytes)
            self._usage.record(self.architecture.value, tenant, audio_seconds=seconds)
            logger.info("usage", extra={"path": self.architecture.value, "audio_seconds": seconds})
        logger.info("realtime turn complete", extra={"latency_ms": latency})
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
    # The validated tenant claim from the auth middleware (VA-15); "default" when auth is
    # off (local, no JWT_SECRET_KEY). Propagated via contextvar because pipelines only see
    # the body model, never the HTTP request.
    return current_tenant()


async def _audio_stream(audio: bytes) -> AsyncIterator[bytes]:
    yield audio
