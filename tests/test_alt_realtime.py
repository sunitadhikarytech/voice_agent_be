"""VA-50 — alternate realtime adapters: Gemini Live + xAI Grok Voice (mocked; no live calls)."""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.config import Settings
from app.providers.base import RealtimeProvider
from app.providers.factory import get_realtime
from app.providers.gemini_live import GeminiLive
from app.providers.grok_realtime import DEFAULT_URL as GROK_URL
from app.providers.grok_realtime import GrokRealtime
from app.providers.openai_realtime import OpenAIRealtime


async def _audio(*chunks: bytes):
    for chunk in chunks:
        yield chunk


# ==================================== Gemini Live ================================================

class FakeLiveSession:
    """Scripted LiveSession: records mic audio, replies with scripted model audio."""

    def __init__(self, replies: list[bytes], *, fail_send: bool = False):
        self._replies = list(replies)
        self._fail_send = fail_send
        self.sent: list[bytes] = []
        self.closed = False

    async def send_audio(self, data: bytes) -> None:
        await asyncio.sleep(0)
        if self._fail_send:
            raise ConnectionError("live send failed")
        self.sent.append(data)

    async def receive(self):
        for reply in self._replies:
            await asyncio.sleep(0)
            yield reply

    async def close(self) -> None:
        self.closed = True


def _live(session: FakeLiveSession) -> GeminiLive:
    async def connect():
        return session

    return GeminiLive("g-key", connect=connect)


def _converse(provider, *chunks: bytes) -> list[bytes]:
    async def run():
        return [c async for c in provider.converse(_audio(*chunks))]

    return asyncio.run(run())


def test_gemini_live_streams_model_audio():
    session = FakeLiveSession([b"model-audio-1", b"model-audio-2"])
    assert _converse(_live(session), b"mic-1", b"mic-2") == [b"model-audio-1", b"model-audio-2"]


def test_gemini_live_forwards_all_mic_audio_and_closes():
    session = FakeLiveSession([b"reply"])
    _converse(_live(session), b"mic-1", b"mic-2", b"mic-3")
    assert session.sent == [b"mic-1", b"mic-2", b"mic-3"]
    assert session.closed is True


def test_gemini_live_empty_model_chunks_are_skipped():
    session = FakeLiveSession([b"", b"real"])
    assert _converse(_live(session), b"mic") == [b"real"]


def test_gemini_live_sender_failure_surfaces_and_closes():
    session = FakeLiveSession([], fail_send=True)
    with pytest.raises(ConnectionError, match="live send failed"):
        _converse(_live(session), b"mic")
    assert session.closed is True


def test_gemini_live_interrupt_is_a_safe_noop():
    # barge-in is native to Gemini Live (server VAD); the seam must accept the call
    asyncio.run(_live(FakeLiveSession([])).interrupt())


def test_gemini_live_satisfies_the_realtime_protocol():
    assert isinstance(GeminiLive("g-key"), RealtimeProvider)


def test_gemini_live_from_settings_and_factory():
    settings = Settings(
        _env_file=None,
        realtime_provider="gemini-live",
        google_api_key="g-key",
        gemini_live_model="gemini-2.0-flash-live-001",
        gemini_live_voice="Puck",
    )
    provider = get_realtime(settings.realtime_provider, settings)
    assert isinstance(provider, GeminiLive)
    assert provider.name == "gemini-live"
    assert "g-key" not in repr(provider)  # key never leaks


# ==================================== Grok (xAI) =================================================

class FakeWsConn:
    """Minimal OpenAI-realtime-protocol server for the inherited machinery."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []
        self.closed = False
        self._i = 0

    async def send(self, data):
        await asyncio.sleep(0)
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if self._i >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._i]
        self._i += 1
        return msg

    async def close(self):
        self.closed = True


def test_grok_is_the_openai_protocol_pointed_at_xai():
    assert issubclass(GrokRealtime, OpenAIRealtime)
    assert GROK_URL.startswith("wss://api.x.ai/")


def test_grok_from_settings_reads_xai_config():
    settings = Settings(
        _env_file=None,
        xai_api_key="xai-key",
        grok_realtime_model="grok-voice",
        grok_voice="ara",
        grok_realtime_url="wss://api.x.ai/v1/realtime",
    )
    provider = GrokRealtime.from_settings(settings)
    assert provider.name == "grok"
    assert provider._url == "wss://api.x.ai/v1/realtime"
    assert provider._model == "grok-voice"
    assert "xai-key" not in repr(provider)
    assert "GrokRealtime" in repr(provider)  # repr names the subclass, not OpenAIRealtime


def test_grok_inherits_the_hardened_session_machinery():
    delta = base64.b64encode(b"grok-audio").decode()
    conn = FakeWsConn([json.dumps({"type": "response.audio.delta", "delta": delta})])

    async def connect():
        return conn

    provider = GrokRealtime("xai-key", connect=connect)
    assert _converse(provider, b"mic-chunk") == [b"grok-audio"]
    # the inherited session.update config frame went out first
    first = json.loads(conn.sent[0])
    assert first["type"] == "session.update"
    assert conn.closed is True


def test_grok_wss_only_guard_applies():
    with pytest.raises(ValueError, match="wss://"):
        GrokRealtime("xai-key", url="ws://api.x.ai/v1/realtime")


def test_grok_factory_builds_from_config():
    settings = Settings(_env_file=None, realtime_provider="grok", xai_api_key="xai-key")
    provider = get_realtime(settings.realtime_provider, settings)
    assert isinstance(provider, GrokRealtime)
