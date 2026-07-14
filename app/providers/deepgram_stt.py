"""Deepgram streaming STT adapter (VA-31).

Streams audio to Deepgram's realtime WebSocket (Nova-3) and yields :class:`TranscriptChunk`s
— interim results, stabilized finals, and an end-of-turn flag at a pause (Deepgram's
``speech_final`` / ``UtteranceEnd``, the signal VA-32 uses for reply timing). A dropped
connection is transparently reconnected (bounded retries) and the stream continues with the
remaining audio.

The transport is injectable (``connect``) so the adapter is fully testable without a socket;
the default opens a real ``websockets`` connection.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Protocol

from app.providers.base import TranscriptChunk

logger = logging.getLogger("app.providers.deepgram")

DEFAULT_URL = "wss://api.deepgram.com/v1/listen"


class Connection(Protocol):
    """Minimal transport contract: send bytes/text, async-iterate JSON messages, close."""

    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[], Awaitable[Connection]]


class SttConnectionError(RuntimeError):
    """Raised when the Deepgram stream cannot be (re)established within the retry budget."""


def _recoverable_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ConnectionError, OSError, asyncio.TimeoutError]
    try:  # add websockets' close exception when the library is available
        import websockets

        errs.append(websockets.exceptions.ConnectionClosed)
    except Exception:  # pragma: no cover - websockets always present in this project
        pass
    return tuple(errs)


_RECOVERABLE = _recoverable_errors()


def parse_message(message: str) -> list[TranscriptChunk]:
    """Map a Deepgram JSON message to zero or more transcript chunks."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return []

    mtype = data.get("type")
    if mtype == "Results":
        alternatives = data.get("channel", {}).get("alternatives", [])
        text = alternatives[0].get("transcript", "") if alternatives else ""
        if not text:
            return []  # empty interim — nothing to surface
        return [
            TranscriptChunk(
                text=text,
                is_final=bool(data.get("is_final", False)),
                is_end_of_turn=bool(data.get("speech_final", False)),
            )
        ]
    if mtype == "UtteranceEnd":
        return [TranscriptChunk(text="", is_final=True, is_end_of_turn=True)]
    return []  # Metadata / SpeechStarted / etc.


class DeepgramStt:
    """SttProvider backed by Deepgram's realtime WebSocket."""

    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        url: str = DEFAULT_URL,
        connect: ConnectFn | None = None,
        max_reconnects: int = 2,
        backoff_base: float = 0.2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._connect = connect or self._default_connect
        self._max_reconnects = max_reconnects
        self._backoff_base = backoff_base

    @classmethod
    def from_settings(cls, settings) -> "DeepgramStt":
        return cls(
            api_key=settings.deepgram_api_key.get_secret_value(),
            model=settings.deepgram_model,
        )

    async def _default_connect(self) -> Connection:
        import websockets

        query = "?model={}&interim_results=true&punctuate=true&smart_format=true".format(
            self._model
        )
        return await websockets.connect(
            self._url + query,
            additional_headers={"Authorization": f"Token {self._api_key}"},
        )

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptChunk]:
        audio_iter = audio.__aiter__()
        attempts = 0
        while True:
            try:
                conn = await self._connect()
            except _RECOVERABLE as exc:
                attempts += 1
                if attempts > self._max_reconnects:
                    raise SttConnectionError("could not connect to Deepgram") from exc
                await asyncio.sleep(self._backoff_base * attempts)
                continue

            try:
                async for chunk in self._run(conn, audio_iter):
                    yield chunk
                return  # audio exhausted and stream finished cleanly
            except _RECOVERABLE as exc:
                attempts += 1
                logger.warning("Deepgram stream dropped (attempt %d): %s", attempts, exc)
                if attempts > self._max_reconnects:
                    raise SttConnectionError("Deepgram stream failed") from exc
                await asyncio.sleep(self._backoff_base * attempts)
                # loop and reconnect, continuing with the remaining audio
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

    async def _run(
        self, conn: Connection, audio_iter: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        sender = asyncio.create_task(self._send_audio(conn, audio_iter))
        try:
            async for message in conn:
                for chunk in parse_message(message):
                    yield chunk
            await sender  # audio fully sent; surface any sender error
        finally:
            if not sender.done():
                sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, *_RECOVERABLE):
                await sender

    async def _send_audio(self, conn: Connection, audio_iter: AsyncIterator[bytes]) -> None:
        async for data in audio_iter:
            await conn.send(data)
        # Tell Deepgram the audio is done so it flushes any pending finals.
        await conn.send(json.dumps({"type": "CloseStream"}))
