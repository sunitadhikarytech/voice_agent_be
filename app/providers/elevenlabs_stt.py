"""ElevenLabs Scribe streaming STT adapter (VA-33) — the alternate STT provider.

Streams audio to ElevenLabs' realtime speech-to-text WebSocket (Scribe) and yields
:class:`TranscriptChunk`s. Select it with ``STT_PROVIDER=elevenlabs`` — proof that swapping
STT vendors is a config change, not a code change (VA-30).

Wire protocol (centralized here so a doc change is a one-line edit):

* client → server: JSON text frames — ``{"audio_chunk": "<base64>"}`` per chunk, then
  ``{"end_of_stream": true}`` to flush pending finals;
* server → client: ``{"type": "partial_transcript"|"final_transcript", "text": ...}``.
  Scribe commits a final at a detected silence, so a ``final_transcript`` doubles as the
  end-of-turn signal VA-32 keys on. Housekeeping messages are ignored.
* auth: ``xi-api-key`` header; model selected by the ``model_id`` query parameter.

Reconnect design mirrors the Deepgram adapter (VA-31 follow-up, PR #14): ONE long-lived pump
task reads the source audio generator into a bounded queue and per-session senders drain the
queue — so per-session teardown never cancels the source generator, and audio left after a
dropped socket flows into the next session instead of being lost. Bounded retries back off
between attempts. The transport is injectable (``connect``) so the adapter is fully testable
without a socket.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Protocol

from app.providers.base import TranscriptChunk

logger = logging.getLogger("app.providers.elevenlabs")

DEFAULT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

# Bounded audio buffer between the single source pump and the per-session sender.
QUEUE_MAXSIZE = 64

# End-of-stream marker for the pump -> sender queue.
_EOS = object()

# Outbound frame that tells Scribe the audio is done (flushes pending finals).
END_OF_STREAM_FRAME = json.dumps({"end_of_stream": True})


class Connection(Protocol):
    """Minimal transport contract: send text, async-iterate JSON messages, close."""

    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[], Awaitable[Connection]]


class SttConnectionError(RuntimeError):
    """Raised when the Scribe stream cannot be (re)established within the retry budget."""


def _recoverable_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ConnectionError, OSError, asyncio.TimeoutError]
    try:  # add websockets' close exception when the library is available
        import websockets

        errs.append(websockets.exceptions.ConnectionClosed)
    except Exception:  # pragma: no cover - websockets always present in this project
        pass
    return tuple(errs)


_RECOVERABLE = _recoverable_errors()


def audio_frame(chunk: bytes) -> str:
    """Encode one audio chunk as the JSON frame Scribe expects."""
    return json.dumps({"audio_chunk": base64.b64encode(chunk).decode("ascii")})


def parse_message(message: str) -> list[TranscriptChunk]:
    """Map a Scribe JSON message to zero or more transcript chunks."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return []

    mtype = data.get("type")
    text = data.get("text", "")
    if mtype == "partial_transcript":
        return [TranscriptChunk(text=text, is_final=False)] if text else []
    if mtype == "final_transcript":
        if not text:
            return []
        # Scribe commits finals at a detected silence — the end-of-turn signal (VA-32).
        return [TranscriptChunk(text=text, is_final=True, is_end_of_turn=True)]
    return []  # session_started / vad events / housekeeping


class ElevenLabsStt:
    """SttProvider backed by ElevenLabs' realtime Scribe WebSocket."""

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "scribe_v2_realtime",
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
    def from_settings(cls, settings) -> "ElevenLabsStt":
        return cls(
            api_key=settings.elevenlabs_api_key.get_secret_value(),
            model=settings.elevenlabs_stt_model,
        )

    async def _default_connect(self) -> Connection:
        import websockets

        return await websockets.connect(
            f"{self._url}?model_id={self._model}",
            additional_headers={"xi-api-key": self._api_key},
        )

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptChunk]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        pump = asyncio.create_task(self._pump(audio, queue))
        attempts = 0
        try:
            while True:
                try:
                    conn = await self._connect()
                except _RECOVERABLE as exc:
                    attempts += 1
                    if attempts > self._max_reconnects:
                        raise SttConnectionError("could not connect to ElevenLabs") from exc
                    await asyncio.sleep(self._backoff_base * attempts)
                    continue

                try:
                    async for chunk in self._run(conn, queue):
                        yield chunk
                    return  # audio exhausted and stream finished cleanly
                except _RECOVERABLE as exc:
                    attempts += 1
                    logger.warning("Scribe stream dropped (attempt %d): %s", attempts, exc)
                    if attempts > self._max_reconnects:
                        raise SttConnectionError("ElevenLabs stream failed") from exc
                    await asyncio.sleep(self._backoff_base * attempts)
                    # loop and reconnect, continuing with the remaining audio
                finally:
                    with contextlib.suppress(Exception):
                        await conn.close()
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def _pump(self, audio: AsyncIterator[bytes], queue: asyncio.Queue) -> None:
        """Single long-lived reader of the source audio generator. Only cancelled when
        ``transcribe()`` itself tears down, so per-session sender cancellation never touches
        the shared generator (cancelling a task parked in ``__anext__`` would finalize it)."""
        try:
            async for chunk in audio:
                await queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface source errors to the sender instead of hanging
            await queue.put(exc)
            return
        await queue.put(_EOS)

    async def _run(
        self, conn: Connection, queue: asyncio.Queue
    ) -> AsyncIterator[TranscriptChunk]:
        """One session: send queued audio while receiving transcripts. A sender failure is
        surfaced promptly (via asyncio.wait) even while the receive side is blocked."""
        sender = asyncio.create_task(self._send_audio(conn, queue))
        recv = conn.__aiter__()
        next_msg: asyncio.Future | None = None
        try:
            while True:
                if next_msg is None:
                    next_msg = asyncio.ensure_future(recv.__anext__())
                wait_for: set[asyncio.Future] = {next_msg}
                if not sender.done():
                    wait_for.add(sender)
                done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)
                if sender in done and not sender.cancelled():
                    sender_exc = sender.exception()
                    if sender_exc is not None:
                        raise sender_exc
                if next_msg in done:
                    try:
                        message = next_msg.result()
                    except StopAsyncIteration:
                        break
                    finally:
                        next_msg = None
                    for chunk in parse_message(message):
                        yield chunk
        finally:
            if next_msg is not None:
                next_msg.cancel()
                # CancelledError is a BaseException: suppress it explicitly so cleanup never
                # clobbers an in-flight (recoverable) stream error.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await next_msg
            if not sender.done():
                sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, *_RECOVERABLE):
                await sender

    async def _send_audio(self, conn: Connection, queue: asyncio.Queue) -> None:
        """Drain the shared queue into this session's socket. On end-of-stream, tell Scribe
        the audio is done (flushes pending finals); the marker is left in the queue so a
        post-reconnect sender re-sends it on the recovered socket."""
        while True:
            item = await queue.get()
            if item is _EOS:
                # Re-mark for any post-reconnect sender before sending (the pump has finished,
                # so the slot freed by get() guarantees put_nowait cannot be full here).
                queue.put_nowait(_EOS)
                await conn.send(END_OF_STREAM_FRAME)
                return
            if isinstance(item, Exception):
                queue.put_nowait(item)  # re-mark for future senders, then surface
                raise item
            await conn.send(audio_frame(item))
