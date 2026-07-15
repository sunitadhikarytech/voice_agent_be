"""VA-44 — ElevenLabs streaming TTS adapter (mocked transport; no live calls)."""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.config import Settings
from app.providers.base import TtsProvider
from app.providers.elevenlabs_tts import (
    BOS_FRAME,
    EOS_FRAME,
    ElevenLabsTts,
    decode_audio,
    text_frame,
)
from app.providers.factory import get_tts


class FakeConn:
    """Scripted stream-input connection: replies with audio after EOS arrives."""

    def __init__(self, messages, *, fail_send: bool = False):
        self._messages = list(messages)
        self._fail_send = fail_send
        self.sent: list[str] = []
        self.closed = False
        self._i = 0

    async def send(self, data):
        await asyncio.sleep(0)
        if self._fail_send:
            raise ConnectionError("simulated send failure")
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        # emit audio only after the client flushed (EOS) — like the real service
        while EOS_FRAME not in self.sent:
            await asyncio.sleep(0)
        if self._i >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._i]
        self._i += 1
        return msg

    async def close(self):
        self.closed = True


def _connect(conn):
    async def connect():
        return conn

    return connect


def audio_msg(data: bytes) -> str:
    return json.dumps({"audio": base64.b64encode(data).decode()})


async def _text(*pieces: str):
    for piece in pieces:
        yield piece


def _synthesize(tts: ElevenLabsTts, *pieces: str) -> list[bytes]:
    async def run():
        return [c async for c in tts.synthesize(_text(*pieces))]

    return asyncio.run(run())


# --- decode --------------------------------------------------------------------------------------

def test_audio_messages_decode_to_bytes():
    assert decode_audio(audio_msg(b"\x00\x01")) == [b"\x00\x01"]


@pytest.mark.parametrize(
    "message",
    ["not json", json.dumps({"isFinal": True}), json.dumps({"audio": ""}), json.dumps({})],
)
def test_noise_and_final_markers_decode_to_nothing(message):
    assert decode_audio(message) == []


# --- protocol ------------------------------------------------------------------------------------

def test_bos_pieces_eos_ordering():
    conn = FakeConn([audio_msg(b"\x01")])
    tts = ElevenLabsTts("xi-key", voice_id="v1", connect=_connect(conn))
    _synthesize(tts, "Hello ", "world.")

    assert conn.sent[0] == BOS_FRAME
    assert conn.sent[1:-1] == [text_frame("Hello "), text_frame("world.")]
    assert conn.sent[-1] == EOS_FRAME
    assert conn.closed is True


def test_empty_pieces_are_not_sent():
    conn = FakeConn([])
    tts = ElevenLabsTts("xi-key", connect=_connect(conn))
    _synthesize(tts, "", "only real text", "")
    assert conn.sent == [BOS_FRAME, text_frame("only real text"), EOS_FRAME]


# --- end to end ----------------------------------------------------------------------------------

def test_streams_audio_chunks_in_order():
    conn = FakeConn([audio_msg(b"\x01\x02"), json.dumps({"isFinal": True}), audio_msg(b"\x03")])
    tts = ElevenLabsTts("xi-key", connect=_connect(conn))
    assert _synthesize(tts, "Some answer.") == [b"\x01\x02", b"\x03"]


def test_sender_failure_surfaces():
    class BrokenSendConn:
        """send() fails; the server closes the stream right away."""

        def __init__(self):
            self.closed = False

        async def send(self, data):
            await asyncio.sleep(0)
            raise ConnectionError("send failed")

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0)
            raise StopAsyncIteration

        async def close(self):
            self.closed = True

    conn = BrokenSendConn()
    tts = ElevenLabsTts("xi-key", connect=_connect(conn))
    with pytest.raises(ConnectionError, match="send failed"):
        _synthesize(tts, "hi")
    assert conn.closed is True  # transport released even on failure


# --- wiring --------------------------------------------------------------------------------------

def test_satisfies_the_tts_protocol():
    assert isinstance(ElevenLabsTts("xi-key"), TtsProvider)


def test_factory_builds_elevenlabs_from_config():
    settings = Settings(
        _env_file=None,
        tts_provider="elevenlabs",
        elevenlabs_api_key="xi-key",
        elevenlabs_voice_id="voice-1",
    )
    tts = get_tts(settings.tts_provider, settings)
    assert isinstance(tts, ElevenLabsTts)
    assert tts.name == "elevenlabs"


def test_from_settings_reads_key_model_and_voice():
    settings = Settings(
        _env_file=None,
        elevenlabs_api_key="xi-key",
        elevenlabs_tts_model="eleven_flash_v2_5",
        elevenlabs_voice_id="voice-9",
    )
    tts = ElevenLabsTts.from_settings(settings)
    assert tts._api_key == "xi-key"
    assert tts._model == "eleven_flash_v2_5"
    assert tts._voice_id == "voice-9"
