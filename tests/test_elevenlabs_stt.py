"""VA-33 — ElevenLabs Scribe streaming STT adapter (mocked transport; no live calls)."""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.config import Settings
from app.providers.base import SttProvider
from app.providers.elevenlabs_stt import (
    END_OF_STREAM_FRAME,
    ElevenLabsStt,
    SttConnectionError,
    audio_frame,
    parse_message,
)
from app.providers.factory import get_stt


# --- fake transport -----------------------------------------------------------------------------

class FakeConn:
    """Scripted Scribe connection (same contract as the Deepgram test double)."""

    def __init__(self, messages, *, drop_after=None, fail_send_at=None, stop_when=None):
        self._messages = list(messages)
        self._drop_after = drop_after
        self._fail_send_at = fail_send_at
        self._stop_when = stop_when
        self.sent: list = []
        self.closed = False
        self._i = 0
        self._sends = 0

    async def send(self, data):
        await asyncio.sleep(0)  # a real socket send suspends
        self._sends += 1
        if self._fail_send_at is not None and self._sends >= self._fail_send_at:
            raise ConnectionError("simulated send failure")
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if self._drop_after is not None and self._i >= self._drop_after:
            raise ConnectionError("simulated drop")
        if self._i >= len(self._messages):
            if self._stop_when is not None:
                while not self._stop_when(self):
                    await asyncio.sleep(0)
            raise StopAsyncIteration
        msg = self._messages[self._i]
        self._i += 1
        return msg

    async def close(self):
        self.closed = True


def connect_seq(*conns):
    it = iter(conns)

    async def _connect():
        try:
            return next(it)
        except StopIteration:
            raise ConnectionError("no more connections")

    return _connect


def partial(text: str) -> str:
    return json.dumps({"type": "partial_transcript", "text": text})


def final(text: str) -> str:
    return json.dumps({"type": "final_transcript", "text": text})


async def _audio(*chunks: bytes):
    for chunk in chunks:
        yield chunk


def _transcribe(stt: ElevenLabsStt, *chunks: bytes):
    async def run():
        return [c async for c in stt.transcribe(_audio(*chunks))]

    return asyncio.run(run())


# --- parse_message -------------------------------------------------------------------------------

def test_partial_maps_to_interim_chunk():
    [chunk] = parse_message(partial("hel"))
    assert chunk.text == "hel" and chunk.is_final is False and chunk.is_end_of_turn is False


def test_final_is_committed_and_ends_the_turn():
    [chunk] = parse_message(final("hello there"))
    assert chunk.text == "hello there"
    assert chunk.is_final is True
    assert chunk.is_end_of_turn is True  # Scribe commits at silence — the VA-32 signal


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        json.dumps({"type": "session_started", "session_id": "x"}),
        json.dumps({"type": "partial_transcript", "text": ""}),  # empty interim
        json.dumps({"type": "final_transcript", "text": ""}),
        json.dumps({"no_type": True}),
    ],
)
def test_noise_and_housekeeping_map_to_nothing(message):
    assert parse_message(message) == []


# --- outbound protocol ---------------------------------------------------------------------------

def test_audio_is_framed_as_base64_json():
    frame = json.loads(audio_frame(b"\x01\x02\x03"))
    assert base64.b64decode(frame["audio_chunk"]) == b"\x01\x02\x03"


def test_sends_all_audio_then_end_of_stream():
    conn = FakeConn(
        [final("done")], stop_when=lambda c: END_OF_STREAM_FRAME in c.sent
    )
    stt = ElevenLabsStt("key", connect=connect_seq(conn))
    _transcribe(stt, b"\x01", b"\x02")

    assert conn.sent[-1] == END_OF_STREAM_FRAME
    audio_frames = [json.loads(f)["audio_chunk"] for f in conn.sent[:-1]]
    assert [base64.b64decode(a) for a in audio_frames] == [b"\x01", b"\x02"]
    assert conn.closed is True


# --- end to end ----------------------------------------------------------------------------------

def test_streams_partials_then_final():
    conn = FakeConn([partial("hel"), partial("hello"), final("hello there")])
    stt = ElevenLabsStt("key", connect=connect_seq(conn))
    chunks = _transcribe(stt, b"\x01")
    assert [c.text for c in chunks] == ["hel", "hello", "hello there"]
    assert [c.is_final for c in chunks] == [False, False, True]
    assert chunks[-1].is_end_of_turn is True


def test_reconnect_resumes_remaining_audio():
    # the first socket dies on its first send; the REMAINING audio must flow into the
    # second session instead of being lost with the source generator (the PR #14 lesson)
    dead = FakeConn([], fail_send_at=1)
    recovered = FakeConn(
        [final("recovered")], stop_when=lambda c: END_OF_STREAM_FRAME in c.sent
    )
    stt = ElevenLabsStt("key", connect=connect_seq(dead, recovered), backoff_base=0.0)
    chunks = _transcribe(stt, b"\x01", b"\x02", b"\x03")

    assert [c.text for c in chunks] == ["recovered"]
    total_audio = b"".join(
        base64.b64decode(json.loads(f)["audio_chunk"])
        for f in recovered.sent
        if f != END_OF_STREAM_FRAME
    )
    # chunk 1 was consumed by the failing send; 2 and 3 survived the reconnect
    assert total_audio == b"\x02\x03"
    assert recovered.sent[-1] == END_OF_STREAM_FRAME


def test_connect_failures_exhaust_retry_budget():
    async def failing_connect():
        raise ConnectionError("refused")

    stt = ElevenLabsStt("key", connect=failing_connect, max_reconnects=1, backoff_base=0.0)
    with pytest.raises(SttConnectionError):
        _transcribe(stt, b"\x01")


# --- wiring --------------------------------------------------------------------------------------

def test_satisfies_the_stt_protocol():
    assert isinstance(ElevenLabsStt("key"), SttProvider)


def test_factory_builds_elevenlabs_from_config():
    settings = Settings(
        _env_file=None, stt_provider="elevenlabs", elevenlabs_api_key="xi-key"
    )
    stt = get_stt(settings.stt_provider, settings)
    assert isinstance(stt, ElevenLabsStt)
    assert stt.name == "elevenlabs"


def test_from_settings_reads_key_and_model():
    settings = Settings(
        _env_file=None, elevenlabs_api_key="xi-key", elevenlabs_stt_model="scribe_v2_realtime"
    )
    stt = ElevenLabsStt.from_settings(settings)
    assert stt._api_key == "xi-key"
    assert stt._model == "scribe_v2_realtime"
