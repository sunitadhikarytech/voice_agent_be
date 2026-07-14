"""VA-43 — Cartesia streaming TTS adapter (mocked transport; no live calls)."""
import asyncio
import base64
import json

from app.config import Settings
from app.providers.base import TtsProvider
from app.providers.cartesia_tts import CartesiaTts, _pop_sentence, decode_audio


class FakeConn:
    """Yields scripted audio messages; records the text requests sent."""

    def __init__(self, audio_chunks):
        self._messages = [
            json.dumps({"type": "chunk", "data": base64.b64encode(a).decode()})
            for a in audio_chunks
        ] + [json.dumps({"type": "done"})]
        self.sent: list = []
        self.closed = False
        self._i = 0

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._i]
        self._i += 1
        return msg

    async def close(self):
        self.closed = True


async def _aiter(items):
    for item in items:
        yield item


async def _collect(agen):
    return [item async for item in agen]


def _sent_transcripts(conn: FakeConn) -> list[str]:
    return [json.loads(s)["transcript"] for s in conn.sent if isinstance(s, str)]


# --- helpers ----------------------------------------------------------------------------

def test_pop_sentence_splits_on_terminator():
    assert _pop_sentence("Hello world. Rest here") == ("Hello world.", "Rest here")
    assert _pop_sentence("no terminator yet") == (None, "no terminator yet")


def test_decode_audio_and_noise():
    msg = json.dumps({"type": "chunk", "data": base64.b64encode(b"\x01\x02").decode()})
    assert decode_audio(msg) == [b"\x01\x02"]
    assert decode_audio(json.dumps({"type": "done"})) == []
    assert decode_audio("not json") == []


# --- streaming --------------------------------------------------------------------------

def test_conforms_to_interface():
    assert isinstance(CartesiaTts(api_key="k"), TtsProvider)


def test_text_in_audio_out():
    conn = FakeConn([b"AAAA", b"BBBB"])
    tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(conn))
    audio = asyncio.run(_collect(tts.synthesize(_aiter(["Hello there."]))))
    assert audio == [b"AAAA", b"BBBB"]
    assert conn.closed


def test_flushes_on_sentence_boundaries():
    conn = FakeConn([b"AAAA"])
    tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(conn))
    # streamed token-by-token; two complete sentences + a trailing fragment
    asyncio.run(_collect(tts.synthesize(_aiter(["Hello ", "world. ", "How are ", "you? ", "Bye"]))))
    assert _sent_transcripts(conn) == ["Hello world.", "How are you?", "Bye"]


def test_request_payload_carries_model_and_voice():
    conn = FakeConn([b"AAAA"])
    tts = CartesiaTts(api_key="k", model="sonic-2", voice_id="voice-123", connect=_connect(conn))
    asyncio.run(_collect(tts.synthesize(_aiter(["Hi."]))))
    payload = json.loads(conn.sent[0])
    assert payload["model_id"] == "sonic-2"
    assert payload["voice"] == {"mode": "id", "id": "voice-123"}


def test_from_settings_reads_config():
    settings = Settings(_env_file=None, cartesia_model="sonic-2", cartesia_voice_id="v9")
    tts = CartesiaTts.from_settings(settings)
    assert tts.name == "cartesia" and tts._model == "sonic-2" and tts._voice_id == "v9"


def _connect(conn):
    async def _c():
        return conn

    return _c
