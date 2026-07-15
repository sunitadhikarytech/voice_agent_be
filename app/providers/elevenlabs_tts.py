"""ElevenLabs streaming TTS adapter (VA-44) — the alternate TTS provider.

Low-latency text-to-speech over ElevenLabs' ``stream-input`` WebSocket: text chunks are
forwarded as the LLM produces them and audio chunks are yielded as they arrive, so the first
audio can start before the full answer is written. Select it with ``TTS_PROVIDER=elevenlabs``
— a config change, no code changes (VA-30).

Wire protocol (the documented stream-input shape):

* client → server: a BOS frame ``{"text": " "}`` opens the stream, each text piece goes as
  ``{"text": "<piece>"}``, and an EOS frame ``{"text": ""}`` flushes remaining audio;
* server → client: ``{"audio": "<base64>"}`` chunks until ``{"isFinal": true}``;
* auth: ``xi-api-key`` header. Voice and model select via the URL path/query; the output
  format is ``pcm_24000`` — the PCM16 @ 24 kHz the reference dashboard (VA-54) plays.

The transport is injectable (``connect``) so the adapter is fully testable without a socket;
the default opens a real ``websockets`` connection.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from typing import AsyncIterator, Awaitable, Callable, Protocol

DEFAULT_URL_TEMPLATE = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

BOS_FRAME = json.dumps({"text": " "})  # opens the stream (a single space, per the docs)
EOS_FRAME = json.dumps({"text": ""})  # flushes and closes generation


class Connection(Protocol):
    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[], Awaitable[Connection]]


class TtsError(RuntimeError):
    """Raised when the ElevenLabs stream fails."""


def text_frame(piece: str) -> str:
    """Encode one text piece as a stream-input frame."""
    return json.dumps({"text": piece})


def decode_audio(message: str) -> list[bytes]:
    """Map a stream-input JSON message to zero or more audio byte chunks."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return []
    audio = data.get("audio")
    if audio:
        return [base64.b64decode(audio)]
    return []  # isFinal / alignment / housekeeping


class ElevenLabsTts:
    """TtsProvider backed by ElevenLabs' stream-input WebSocket."""

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "eleven_flash_v2_5",
        voice_id: str = "",
        sample_rate: int = 24000,
        url_template: str = DEFAULT_URL_TEMPLATE,
        connect: ConnectFn | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._url_template = url_template
        self._connect = connect or self._default_connect

    @classmethod
    def from_settings(cls, settings) -> "ElevenLabsTts":
        return cls(
            api_key=settings.elevenlabs_api_key.get_secret_value(),
            model=settings.elevenlabs_tts_model,
            voice_id=settings.elevenlabs_voice_id,
        )

    async def _default_connect(self) -> Connection:
        import websockets

        url = self._url_template.format(voice_id=self._voice_id)
        query = f"?model_id={self._model}&output_format=pcm_{self._sample_rate}"
        return await websockets.connect(
            url + query, additional_headers={"xi-api-key": self._api_key}
        )

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        conn = await self._connect()
        try:
            sender = asyncio.create_task(self._send_text(conn, text))
            try:
                async for message in conn:
                    for audio in decode_audio(message):
                        yield audio
                await sender  # surface any sender error once the stream ends
            finally:
                if not sender.done():
                    sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError, OSError):
                    await sender
        finally:
            with contextlib.suppress(Exception):
                await conn.close()

    async def _send_text(self, conn: Connection, text: AsyncIterator[str]) -> None:
        """BOS, then each text piece as it becomes available, then the EOS flush. ElevenLabs
        buffers and schedules generation internally, so pieces stream straight through."""
        await conn.send(BOS_FRAME)
        async for piece in text:
            if piece:
                await conn.send(text_frame(piece))
        await conn.send(EOS_FRAME)
