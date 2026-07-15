"""Traditional pipeline: STT → LLM → TTS end to end (VA-45).

Orchestrates the three stages into a single streaming turn: transcribe the input (Deepgram),
generate a grounded, tool-aware answer (Gemini over the full-document context), and synthesize
speech (Cartesia) — emitting the shared SSE events throughout and driving the per-turn state
machine. Grounding (VA-37), tools (VA-38) and the document cache (VA-36) are wired onto the LLM
once at construction; conversation memory (VA-41) is rebuilt each turn from the session (VA-40).

Stages are composed directly (async generators) rather than via a heavyweight framework, so
the whole turn is deterministic and testable with the mock providers.
"""
from __future__ import annotations

import base64
import time
from typing import AsyncIterator, Callable

from app.context import ground_llm
from app.context.loader import DocumentContext
from app.dispatch import Architecture
from app.pipelines.base import BasePipeline
from app.providers.base import LlmProvider, SttProvider, TranscriptChunk, TtsProvider
from app.session import ConversationMemory, SessionStore, TurnState, TurnStateMachine
from app.streaming.events import (
    AnswerDelta,
    AnySSEEvent,
    AudioChunk,
    Done,
    TranscriptFinal,
    TranscriptPartial,
)
from app.streaming.schemas import AudioInput, TextInput, VoiceTurnRequest, VoiceTurnResult
from app.tools import ToolRegistry


async def _once(item: str) -> AsyncIterator[str]:
    yield item


class TraditionalPipeline(BasePipeline):
    """The document-grounded, tool-capable slow path."""

    architecture = Architecture.TRADITIONAL

    def __init__(
        self,
        stt: SttProvider,
        llm: LlmProvider,
        tts: TtsProvider,
        *,
        session_store: SessionStore | None = None,
        memory: ConversationMemory | None = None,
        tools: ToolRegistry | None = None,
        document: DocumentContext | None = None,
        state_factory: Callable[[], TurnStateMachine] = TurnStateMachine,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._sessions = session_store or SessionStore()
        self._memory = memory or ConversationMemory()
        self._state_factory = state_factory
        self._clock = clock

        # Wire grounding + tools onto the LLM once (the grounded prompt + doc are cached by VA-36).
        if document is not None:
            ground_llm(self._llm, document)
        if tools is not None:
            # exposed to the model; the model decides when to call them (dispatch in a follow-up)
            self._llm.tools = tools.declarations()

    # --- streaming delivery -------------------------------------------------------------

    async def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        session = self._sessions.resolve(_tenant_of(request), request.session_id)
        state = self._state_factory()
        started = self._clock()
        latency: dict[str, float] = {}

        # 1) Listen / transcribe.
        state.transition(TurnState.LISTENING)
        transcript = ""
        async for chunk in self._transcribe(request):
            if not chunk.text:
                continue
            if chunk.is_final:
                transcript = chunk.text
                yield TranscriptFinal(text=chunk.text)
            else:
                yield TranscriptPartial(text=chunk.text)
        latency["stt_ms"] = self._elapsed_ms(started)
        session.add_turn("user", transcript)

        # 2) Think / generate a grounded answer over the rolling conversation memory.
        state.transition(TurnState.THINKING)
        prompt = self._memory.build(session.turns)
        answer_parts: list[str] = []
        llm_started = self._clock()
        async for token in self._llm.generate(prompt):
            answer_parts.append(token)
            yield AnswerDelta(text=token)
        answer = "".join(answer_parts)
        latency["llm_ms"] = self._elapsed_ms(llm_started)
        session.add_turn("agent", answer)

        # 3) Speak / synthesize.
        state.transition(TurnState.SPEAKING)
        seq = 0
        first_audio: float | None = None
        async for audio in self._tts.synthesize(_once(answer)):
            if first_audio is None:
                first_audio = self._elapsed_ms(started)
            yield AudioChunk(audio_b64=base64.b64encode(audio).decode("ascii"), seq=seq)
            seq += 1
        if first_audio is not None:
            latency["first_audio_ms"] = first_audio

        state.transition(TurnState.IDLE)
        yield Done(session_id=session.session_id, latency_ms=latency)

    # --- complete delivery --------------------------------------------------------------

    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        transcript = ""
        answer_parts: list[str] = []
        latency: dict[str, float] = {}
        session_id = request.session_id
        async for event in self.stream(request):
            if isinstance(event, TranscriptFinal):
                transcript = event.text
            elif isinstance(event, AnswerDelta):
                answer_parts.append(event.text)
            elif isinstance(event, Done):
                latency = event.latency_ms
                session_id = event.session_id
        return VoiceTurnResult(
            session_id=session_id,
            transcript=transcript,
            answer_text="".join(answer_parts),
            audio_url=None,  # streamed as audio.chunk; persisted-audio URL is a later concern
            tools_called=[],
            latency_ms=latency,
        )

    # --- helpers ------------------------------------------------------------------------

    async def _transcribe(self, request: VoiceTurnRequest) -> AsyncIterator[TranscriptChunk]:
        """Text input is already the transcript; audio input runs through the STT adapter."""
        value = request.input
        if isinstance(value, TextInput):
            yield TranscriptChunk(text=value.text, is_final=True, is_end_of_turn=True)
            return
        assert isinstance(value, AudioInput)
        audio_bytes = base64.b64decode(value.audio_b64)
        async for chunk in self._stt.transcribe(_audio_stream(audio_bytes)):
            yield chunk

    def _elapsed_ms(self, since: float) -> float:
        return round((self._clock() - since) * 1000, 3)


def _tenant_of(request: VoiceTurnRequest) -> str:
    # The auth middleware (VA-15) attaches the validated tenant; default until then.
    return "default"


async def _audio_stream(audio: bytes) -> AsyncIterator[bytes]:
    yield audio
