"""Deepgram streaming STT adapter (VA-31).

Streams audio to Deepgram's realtime WebSocket (Nova-3) and yields :class:`TranscriptChunk`s
— interim results, stabilized finals, and an end-of-turn flag at a pause (Deepgram's
``speech_final`` / ``UtteranceEnd``, the signal VA-32 uses for reply timing).

Reconnect design (VA-31 follow-up): the source audio generator is pumped by ONE long-lived
task into a bounded queue, and each session's sender drains the queue. Per-session teardown
therefore never cancels the source generator itself, so after a dropped socket the remaining
audio flows into the next session instead of being lost (cancelling a task parked in
``audio.__anext__()`` would otherwise finalize the shared generator). Bounded retries back
off between attempts.

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

# Bounded audio buffer between the single source pump and the per-session sender.
QUEUE_MAXSIZE = 64

# End-of-stream marker for the pump -> sender queue.
_EOS = object()


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
                        raise SttConnectionError("could not connect to Deepgram") from exc
                    await asyncio.sleep(self._backoff_base * attempts)
                    continue

                try:
                    async for chunk in self._run(conn, queue):
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
        """Drain the shared queue into this session's socket. On end-of-stream, tell Deepgram
        the audio is done (``CloseStream``) so it flushes pending finals; the marker is left in
        the queue so a post-reconnect sender re-sends it on the recovered socket."""
        while True:
            item = await queue.get()
            if item is _EOS:
                # Re-mark for any post-reconnect sender before sending (the pump has finished,
                # so the slot freed by get() guarantees put_nowait cannot be full here).
                queue.put_nowait(_EOS)
                await conn.send(json.dumps({"type": "CloseStream"}))
                return
            if isinstance(item, Exception):
                queue.put_nowait(item)  # re-mark for future senders, then surface
                raise item
            await conn.send(item)
