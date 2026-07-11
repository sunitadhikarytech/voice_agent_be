"""Provider adapter interfaces (STT / LLM / TTS).

Every speech-to-text, language-model and text-to-speech provider sits behind one of these
small interfaces, so a provider can be swapped by configuration rather than code. VA-01
defines the contracts; concrete adapters arrive in their own tickets (Deepgram VA-31,
Gemini VA-34, Cartesia VA-43, alternates VA-33/VA-44, etc.).
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class SttProvider(Protocol):
    """Streaming speech-to-text."""

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Consume audio chunks and yield transcript text (partials then finals)."""
        ...


@runtime_checkable
class LlmProvider(Protocol):
    """Language model that streams a text answer for a transcript."""

    async def generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Yield answer tokens for the given prompt/context."""
        ...


@runtime_checkable
class TtsProvider(Protocol):
    """Streaming text-to-speech."""

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Consume text and yield audio chunks."""
        ...
