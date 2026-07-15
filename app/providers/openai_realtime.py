"""OpenAI Realtime voice-to-voice adapter (VA-46) — the fast path's core.

Opens a persistent WebSocket to the OpenAI Realtime API (beta ``realtime=v1`` protocol),
streams mic audio in and model audio out over one session, and manages the connection
lifecycle. Server-side VAD drives turn-taking; ``interrupt()`` cancels the in-flight response
for barge-in (wired by VA-47).

Reconnect design: the mic generator is pumped by ONE long-lived task into a bounded queue,
and each session's sender drains the queue. Per-session teardown therefore never cancels the
mic generator itself, so after a dropped socket the remaining audio flows into the next
session instead of being lost (cancelling a task parked in ``audio_in.__anext__()`` would
otherwise finalize the shared generator).

The transport is injectable (``connect``) so the adapter is fully testable without a socket;
the default opens a real ``websockets`` connection.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Protocol

logger = logging.getLogger("app.providers.openai_realtime")

DEFAULT_URL = "wss://api.openai.com/v1/realtime"

# Bounded mic buffer: ~64 chunks of 20 ms PCM16 @ 24 kHz ≈ 1.3 s of audio.
QUEUE_MAXSIZE = 64

# End-of-stream marker for the pump -> sender queue.
_EOS = object()


class Connection(Protocol):
    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[], Awaitable[Connection]]


class RealtimeError(RuntimeError):
    """Raised when the realtime session cannot be sustained within the retry budget."""


def _recoverable_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ConnectionError, OSError, asyncio.TimeoutError]
    try:
        import websockets

        errs.append(websockets.exceptions.ConnectionClosed)
    except Exception:  # pragma: no cover - websockets always present in this project
        pass
    return tuple(errs)


_RECOVERABLE = _recoverable_errors()


def parse_audio(message: str) -> list[bytes]:
    """Extract output audio (PCM bytes) from an OpenAI Realtime server event.

    Defensive against untrusted input: any malformed frame (non-JSON, non-object JSON, or
    invalid base64) is skipped rather than crashing the session.
    """
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    if data.get("type") == "response.audio.delta" and data.get("delta"):
        try:
            return [base64.b64decode(data["delta"])]
        except (ValueError, TypeError):
            return []
    return []  # session.*, response.done, transcript events, etc.


class OpenAIRealtime:
    """RealtimeProvider backed by the OpenAI Realtime WebSocket API."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        url: str = DEFAULT_URL,
        connect: ConnectFn | None = None,
        max_reconnects: int = 2,
        backoff_base: float = 0.2,
    ) -> None:
        if not url.startswith("wss://"):
            # The Bearer token must never travel over a cleartext socket.
            raise ValueError("realtime url must use wss://")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._url = url
        self._connect = connect or self._default_connect
        self._max_reconnects = max_reconnects
        self._backoff_base = backoff_base
        self._active: Connection | None = None  # current session, for interrupt()

    def __repr__(self) -> str:  # never expose the API key
        return f"{type(self).__name__}(model={self._model!r}, voice={self._voice!r})"

    @classmethod
    def from_settings(cls, settings) -> "OpenAIRealtime":
        return cls(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_realtime_model,
            voice=settings.openai_voice,
        )

    async def _default_connect(self) -> Connection:
        import websockets

        return await websockets.connect(
            f"{self._url}?model={self._model}",
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

    def _session_update(self) -> str:
        return json.dumps(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": self._voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                },
            }
        )

    async def interrupt(self) -> None:
        """Cancel the in-flight response and clear the input buffer (barge-in, VA-47).

        An interrupt that lands during a reconnect window (no active session) is dropped by
        design; VA-47 owns barge-in hardening.
        """
        conn = self._active
        if conn is None:
            return
        with contextlib.suppress(*_RECOVERABLE):
            await conn.send(json.dumps({"type": "response.cancel"}))
            await conn.send(json.dumps({"type": "input_audio_buffer.clear"}))

    async def converse(self, audio_in: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        pump = asyncio.create_task(self._pump(audio_in, queue))
        attempts = 0
        try:
            while True:
                try:
                    conn = await self._connect()
                except _RECOVERABLE as exc:
                    attempts += 1
                    if attempts > self._max_reconnects:
                        raise RealtimeError("could not connect to OpenAI Realtime") from exc
                    await asyncio.sleep(self._backoff_base * attempts)
                    continue

                self._active = conn
                try:
                    # Sessions are per-socket: (re)establish config on every connection.
                    await conn.send(self._session_update())
                    async for audio in self._run(conn, queue):
                        attempts = 0  # healthy output -> reset the per-incident retry budget
                        yield audio
                    return  # session finished cleanly
                except _RECOVERABLE as exc:
                    attempts += 1
                    logger.warning("Realtime session dropped (attempt %d): %s", attempts, exc)
                    if attempts > self._max_reconnects:
                        raise RealtimeError("OpenAI Realtime session failed") from exc
                    await asyncio.sleep(self._backoff_base * attempts)
                finally:
                    self._active = None
                    with contextlib.suppress(Exception):
                        await conn.close()
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def _pump(self, audio_in: AsyncIterator[bytes], queue: asyncio.Queue) -> None:
        """Single long-lived reader of the mic generator. Only cancelled when converse()
        itself tears down, so per-session sender cancellation never touches ``audio_in``."""
        try:
            async for chunk in audio_in:
                await queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface mic errors to the sender instead of hanging
            await queue.put(exc)
            return
        await queue.put(_EOS)

    async def _run(self, conn: Connection, queue: asyncio.Queue) -> AsyncIterator[bytes]:
        """One session: send queued audio while receiving model audio. A sender failure is
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
                    for audio in parse_audio(message):
                        yield audio
        finally:
            if next_msg is not None:
                next_msg.cancel()
                # CancelledError is a BaseException: suppress it explicitly so cleanup never
                # clobbers an in-flight (recoverable) session error.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await next_msg
            if not sender.done():
                sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, *_RECOVERABLE):
                await sender

    async def _send_audio(self, conn: Connection, queue: asyncio.Queue) -> None:
        """Drain the shared queue into this session's socket. Server VAD owns turn commits,
        so no manual input_audio_buffer.commit is sent."""
        while True:
            item = await queue.get()
            if item is _EOS:
                # Leave the marker for any post-reconnect sender (the pump has finished, so
                # the slot freed by get() guarantees put_nowait cannot be full here).
                queue.put_nowait(_EOS)
                return
            if isinstance(item, Exception):
                queue.put_nowait(item)  # re-mark for future senders, then surface
                raise item
            await conn.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(item).decode("ascii"),
                    }
                )
            )
