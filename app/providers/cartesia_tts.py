"""Cartesia streaming TTS adapter (VA-43).

Low-latency text-to-speech: streams text to Cartesia's realtime WebSocket (Sonic) and yields
audio chunks as they arrive, so the first audio can start before the full answer is written.
Text is flushed on sentence boundaries for natural prosody. Behind the ``TtsProvider``
interface (VA-30).

Context protocol (live-verified 2026-07-16): every request frame must carry a ``context_id``
— the API rejects frames without one (400). One synthesize() call = one context: each
sentence is sent as an input continuation (``continue: true``) and an empty-transcript
finalizer (``continue: false``) closes the context, after which the server emits its ``done``
frame. Error frames raise :class:`TtsError` and ``done`` ends the receive loop — a failed
context must fail the turn, not hang it.

The transport is injectable (``connect``) so the adapter is fully testable without a socket;
the default opens a real ``websockets`` connection.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import uuid
from typing import AsyncIterator, Awaitable, Callable, Protocol

DEFAULT_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2024-11-13"

# Split off a complete sentence (up to and including . ! ? …) when one is available.
_SENTENCE_RE = re.compile(r"^(.*?[.!?…])(\s+)(.*)$", re.DOTALL)


class Connection(Protocol):
    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[], Awaitable[Connection]]


class TtsError(RuntimeError):
    """Raised when the Cartesia stream fails."""


def _pop_sentence(buffer: str) -> tuple[str | None, str]:
    """Return (complete_sentence, remainder). Sentence is None when the buffer has no
    terminated sentence yet."""
    match = _SENTENCE_RE.match(buffer)
    if not match:
        return None, buffer
    return match.group(1), match.group(3)


def decode_audio(message: str) -> list[bytes]:
    """Map a Cartesia JSON message to zero or more audio byte chunks."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return []
    if data.get("type") == "chunk" and data.get("data"):
        return [base64.b64decode(data["data"])]
    return []  # "done" / "timestamps" / errors handled by parse_frame


def parse_frame(message: str) -> tuple[list[bytes], bool]:
    """Interpret one server frame: ``(audio_chunks, stream_done)``.

    An ``error`` frame raises :class:`TtsError` — before this the adapter silently ignored
    errors and hung waiting for audio that would never come (found live, 2026-07-16). A
    ``done`` frame (or a frame flagged ``done: true``) ends the stream.
    """
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return [], False
    if not isinstance(data, dict):
        return [], False
    if data.get("type") == "error":
        raise TtsError(
            f"Cartesia stream error (status {data.get('status_code')}): {data.get('error')}"
        )
    return decode_audio(message), data.get("type") == "done" or data.get("done") is True


class CartesiaTts:
    """TtsProvider backed by Cartesia's realtime WebSocket."""

    name = "cartesia"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "sonic-2",
        voice_id: str = "",
        sample_rate: int = 24000,
        url: str = DEFAULT_URL,
        connect: ConnectFn | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._url = url
        self._connect = connect or self._default_connect

    @classmethod
    def from_settings(cls, settings) -> "CartesiaTts":
        return cls(
            api_key=settings.cartesia_api_key.get_secret_value(),
            model=settings.cartesia_model,
            voice_id=settings.cartesia_voice_id,
        )

    async def _default_connect(self) -> Connection:
        import websockets

        return await websockets.connect(
            f"{self._url}?api_key={self._api_key}&cartesia_version={CARTESIA_VERSION}"
        )

    def _request(self, text: str, *, context_id: str, cont: bool) -> str:
        return json.dumps(
            {
                "model_id": self._model,
                "transcript": text,
                "voice": {"mode": "id", "id": self._voice_id},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self._sample_rate,
                },
                # Required by the API (frames without one are rejected with a 400) and what
                # keeps sentence continuations ordered within the turn.
                "context_id": context_id,
                "continue": cont,
            }
        )

    async def synthesize(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        context_id = uuid.uuid4().hex  # one context per turn
        conn = await self._connect()
        try:
            sender = asyncio.create_task(self._send_text(conn, text, context_id))
            try:
                async for message in conn:
                    audio_chunks, stream_done = parse_frame(message)  # raises on error frames
                    for audio in audio_chunks:
                        yield audio
                    if stream_done:
                        break
                await sender  # surface any sender error once the stream ends
            finally:
                if not sender.done():
                    sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError, OSError):
                    await sender
        finally:
            with contextlib.suppress(Exception):
                await conn.close()

    async def _send_text(self, conn: Connection, text: AsyncIterator[str], context_id: str) -> None:
        buffer = ""
        async for chunk in text:
            buffer += chunk
            # Flush each complete sentence as soon as it's available (low latency, natural prosody).
            while True:
                sentence, buffer = _pop_sentence(buffer)
                if sentence is None:
                    break
                await conn.send(self._request(sentence, context_id=context_id, cont=True))
        if buffer.strip():
            await conn.send(self._request(buffer.strip(), context_id=context_id, cont=True))
        # Close the context: the server flushes remaining audio and emits its done frame.
        await conn.send(self._request("", context_id=context_id, cont=False))
