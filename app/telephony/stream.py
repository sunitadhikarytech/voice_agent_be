"""Twilio Media Streams protocol transport (VA-72).

Wraps a WebSocket connection and speaks Twilio's Media Streams JSON protocol so the rest of
the code deals in raw μ-law audio, not wire framing. Inbound messages:

* ``connected`` — handshake (ignored)
* ``start`` — carries ``streamSid``/``callSid`` (captured; needed to send audio back)
* ``media`` — a base64 μ-law frame of caller audio
* ``mark`` — playback checkpoint we requested (surfaced so the bridge can track completion)
* ``stop`` — the call ended

Outbound we send ``media`` (audio to play), ``mark`` (checkpoint), and ``clear`` (flush
buffered audio — used for barge-in). The socket is injected, so tests drive it with a fake
that scripts inbound frames and records what we send.
"""
from __future__ import annotations

import base64
import json
from typing import AsyncIterator, Protocol

from fastapi import WebSocketDisconnect


class WebSocketLike(Protocol):
    async def receive_text(self) -> str: ...
    async def send_text(self, data: str) -> None: ...


class TwilioMediaStream:
    """Bidirectional Twilio Media Stream over one WebSocket (one phone call)."""

    def __init__(self, ws: WebSocketLike) -> None:
        self._ws = ws
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.marks: list[str] = []

    async def wait_for_start(self) -> bool:
        """Read frames until Twilio's ``start`` (capturing ``streamSid``), so the agent knows
        where to send audio before it speaks. Returns ``False`` if the call ends first."""
        while self.stream_sid is None:
            try:
                raw = await self._ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                return False
            message = json.loads(raw)
            if message.get("event") == "start":
                start = message.get("start", {})
                self.stream_sid = start.get("streamSid") or message.get("streamSid")
                self.call_sid = start.get("callSid")
                return True
            if message.get("event") == "stop":
                return False
        return True

    async def inbound_audio(self) -> AsyncIterator[bytes]:
        """Yield caller audio as μ-law frames until the call stops or the socket closes.

        ``start`` is captured (for the streamSid needed to talk back); ``stop`` and a
        disconnect both cleanly end iteration.
        """
        while True:
            try:
                raw = await self._ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                return
            message = json.loads(raw)
            event = message.get("event")
            if event == "start":
                start = message.get("start", {})
                self.stream_sid = start.get("streamSid") or message.get("streamSid")
                self.call_sid = start.get("callSid")
            elif event == "media":
                payload = message.get("media", {}).get("payload")
                if payload:
                    yield base64.b64decode(payload)
            elif event == "mark":
                name = message.get("mark", {}).get("name")
                if name:
                    self.marks.append(name)
            elif event == "stop":
                return

    async def send_audio(self, mulaw: bytes) -> None:
        """Send one μ-law frame for Twilio to play back to the caller."""
        if not mulaw:
            return
        await self._ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(mulaw).decode("ascii")},
                }
            )
        )

    async def send_mark(self, name: str) -> None:
        """Mark a point in the outbound audio; Twilio echoes it back once played."""
        await self._ws.send_text(
            json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})
        )

    async def clear(self) -> None:
        """Drop any audio Twilio has buffered but not yet played (barge-in)."""
        await self._ws.send_text(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
