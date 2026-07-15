"""Deterministic mock providers (VA-30).

These satisfy the STT/LLM/TTS interfaces without any network calls, so pipelines and tests
can run end-to-end offline. Registered under the name ``"mock"`` in ``app.providers.factory``.
"""
from __future__ import annotations

from typing import AsyncIterator

from app.providers.base import TranscriptChunk


class MockStt:
    """Drains the audio stream and emits a canned partial then final transcript."""

    name = "mock"

    def __init__(self, transcript: str = "mock transcript") -> None:
        self._transcript = transcript

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptChunk]:
        async for _chunk in audio:
            pass
        yield TranscriptChunk(text=self._transcript, is_final=False)
        yield TranscriptChunk(text=self._transcript, is_final=True, is_end_of_turn=True)


class MockLlm:
    """Streams a fixed set of answer tokens."""

    name = "mock"

    def __init__(self, tokens: tuple[str, ...] = ("mock ", "answer")) -> None:
        self._tokens = tokens
        # Mirror the settable attributes the pipeline and grounding (VA-37) mutate on a real
        # LLM adapter (see GeminiLlm), so grounding works with the mock when a document is
        # loaded instead of raising AttributeError.
        self.system_prompt = "You are a helpful voice assistant."
        self.document_context: str | None = None
        self.tools: list | None = None

    async def generate(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


class MockTts:
    """Emits one audio chunk per text chunk received."""

    name = "mock"

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for chunk in text:
            yield chunk.encode("utf-8")


class MockRealtime:
    """Voice-to-voice mock: echoes each input audio chunk back as output."""

    name = "mock"

    def __init__(self) -> None:
        self.interrupts = 0

    async def converse(self, audio_in: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for chunk in audio_in:
            yield b"out:" + chunk

    async def interrupt(self) -> None:
        self.interrupts += 1
