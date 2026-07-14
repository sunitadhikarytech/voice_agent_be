"""Provider adapter interfaces (STT / LLM / TTS) — VA-30.

Every speech-to-text, language-model and text-to-speech provider implements one of these
small structural interfaces, so a provider can be swapped by configuration rather than code
(see ``app.providers.factory``). The streaming methods are async generators, so they are typed
as callables returning an ``AsyncIterator``. Concrete adapters arrive in their own tickets
(Deepgram VA-31, Gemini VA-34, Cartesia VA-43, realtime VA-46, alternates VA-33/44/50); a
deterministic mock lives in ``app.providers.mock``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass(slots=True)
class TranscriptChunk:
    """A unit of streaming transcription.

    ``is_final`` distinguishes a stabilized transcript from an interim guess; ``is_end_of_turn``
    marks that the speaker has paused (the signal that drives reply timing in VA-32). These map
    onto the ``transcript.partial`` / ``transcript.final`` SSE events (VA-20).
    """

    text: str
    is_final: bool = False
    is_end_of_turn: bool = False


@runtime_checkable
class SttProvider(Protocol):
    """Streaming speech-to-text."""

    name: str

    def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptChunk]:
        """Consume audio chunks and yield transcription as it is recognized."""
        ...


@runtime_checkable
class LlmProvider(Protocol):
    """Language model that streams a text answer."""

    name: str

    def generate(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield answer tokens for the prompt (with an optional system instruction)."""
        ...


@runtime_checkable
class TtsProvider(Protocol):
    """Streaming text-to-speech."""

    name: str

    def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Consume text and yield audio chunks."""
        ...
