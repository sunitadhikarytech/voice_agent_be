"""Gemini Live realtime adapter (VA-50) — alternate voice-to-voice provider.

Bridges Google's Live API (bidirectional audio over one session, via the ``google-genai``
SDK already used by the VA-34 LLM adapter) into the ``RealtimeProvider`` interface. Select
it with ``REALTIME_PROVIDER=gemini-live`` — a config change, no code changes (VA-30).

Design choices, deliberately different from the OpenAI adapter:

* **One live session per ``converse()`` call, no reconnect-with-resume.** A dropped session
  surfaces as an error, and the VA-49 fallback wrapper already degrades that turn to the
  traditional pipeline — vendor-specific resume logic isn't duplicated here.
* **``interrupt()`` is a logged no-op at this seam.** Gemini Live runs its own server-side
  VAD and interrupts generation natively when the user talks over the reply; the barge-in
  behaviour is provided by the service rather than a client cancel frame.
* The session surface is injectable (``connect``) behind a minimal :class:`LiveSession`
  protocol, so the adapter is fully testable without the SDK or a network. The default
  adapts the real SDK session (``client.aio.live.connect``).

Audio in is PCM16 mic audio; audio out is the model's PCM stream (24 kHz), matching the
pipeline contract the dashboard plays.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncIterator, Awaitable, Callable, Protocol

logger = logging.getLogger("app.providers.gemini_live")

DEFAULT_MODEL = "gemini-2.0-flash-live-001"
INPUT_MIME = "audio/pcm;rate=16000"


class LiveSession(Protocol):
    """Minimal live-session contract: push mic audio, iterate model audio, close."""

    async def send_audio(self, data: bytes) -> None: ...
    def receive(self) -> AsyncIterator[bytes]: ...
    async def close(self) -> None: ...


SessionFactory = Callable[[], Awaitable[LiveSession]]


class GeminiLiveError(RuntimeError):
    """Raised when a live session cannot be established."""


class GeminiLive:
    """RealtimeProvider backed by the Gemini Live API."""

    name = "gemini-live"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = "Puck",
        connect: SessionFactory | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._connect = connect or self._default_connect

    def __repr__(self) -> str:  # never expose the API key
        return f"GeminiLive(model={self._model!r}, voice={self._voice!r})"

    @classmethod
    def from_settings(cls, settings) -> "GeminiLive":
        return cls(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.gemini_live_model,
            voice=settings.gemini_live_voice,
        )

    async def _default_connect(self) -> LiveSession:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        ctx = client.aio.live.connect(
            model=self._model,
            config=types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._voice)
                    )
                ),
            ),
        )
        try:
            session = await ctx.__aenter__()
        except Exception as exc:  # pragma: no cover - SDK/network specific
            raise GeminiLiveError("could not open a Gemini Live session") from exc
        return _SdkLiveSession(ctx, session)

    async def converse(self, audio_in: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        session = await self._connect()
        try:
            sender = asyncio.create_task(self._send(session, audio_in))
            try:
                async for audio in session.receive():
                    if audio:
                        yield audio
                await sender  # surface any sender error once the stream ends
            finally:
                if not sender.done():
                    sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError, OSError):
                    await sender
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    async def _send(self, session: LiveSession, audio_in: AsyncIterator[bytes]) -> None:
        async for chunk in audio_in:
            await session.send_audio(chunk)

    async def interrupt(self) -> None:
        """Barge-in is native to Gemini Live: its server-side VAD interrupts generation
        when the user talks over the reply — there is nothing to cancel at this seam."""
        logger.info("interrupt requested; Gemini Live handles barge-in via server VAD")


class _SdkLiveSession:
    """Adapts the ``google-genai`` live session to the :class:`LiveSession` protocol."""

    def __init__(self, ctx, session) -> None:
        self._ctx = ctx
        self._session = session

    async def send_audio(self, data: bytes) -> None:
        from google.genai import types

        await self._session.send_realtime_input(
            audio=types.Blob(data=data, mime_type=INPUT_MIME)
        )

    async def receive(self) -> AsyncIterator[bytes]:
        async for message in self._session.receive():
            data = getattr(message, "data", None)
            if data:
                yield data

    async def close(self) -> None:
        await self._ctx.__aexit__(None, None, None)
