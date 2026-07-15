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
    # every non-final frame is a spoken transcript; the empty finalizer closes the context
    return [
        json.loads(s)["transcript"]
        for s in conn.sent
        if isinstance(s, str) and json.loads(s)["continue"]
    ]


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


# --- context protocol + error frames (VA-43 follow-up, found live) -------------------------------

def test_every_frame_carries_the_same_context_id_and_a_finalizer():
    conn = FakeConn([b"AAAA"])
    tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(conn))
    asyncio.run(_collect(tts.synthesize(_aiter(["One. ", "Two. ", "tail"]))))

    frames = [json.loads(s) for s in conn.sent]
    context_ids = {f["context_id"] for f in frames}
    assert len(context_ids) == 1 and next(iter(context_ids))  # one non-empty context per turn
    assert [f["continue"] for f in frames] == [True, True, True, False]
    assert frames[-1]["transcript"] == ""  # the empty finalizer closes the context


def test_each_turn_gets_its_own_context():
    tts_ids = []
    for _ in range(2):
        conn = FakeConn([b"AAAA"])
        tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(conn))
        asyncio.run(_collect(tts.synthesize(_aiter(["Hi."]))))
        tts_ids.append(json.loads(conn.sent[0])["context_id"])
    assert tts_ids[0] != tts_ids[1]


def test_error_frame_raises_instead_of_hanging():
    import pytest

    from app.providers.cartesia_tts import TtsError

    class ErrorConn(FakeConn):
        def __init__(self):
            super().__init__([])
            # server rejects the context and then goes silent — must raise, not hang
            self._messages = [
                json.dumps({"type": "error", "status_code": 400, "done": True,
                            "error": "context_id is invalid"})
            ]

    tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(ErrorConn()))
    with pytest.raises(TtsError, match="status 400.*context_id"):
        asyncio.run(_collect(tts.synthesize(_aiter(["Hi."]))))


def test_done_frame_ends_the_stream_even_if_the_socket_stays_open():
    class OpenSocketConn(FakeConn):
        """Serves audio + done, then blocks forever — like a real keep-alive socket."""

        async def __anext__(self):
            if self._i >= len(self._messages):
                await asyncio.Event().wait()  # never set: hangs unless done is honoured
            msg = self._messages[self._i]
            self._i += 1
            return msg

    conn = OpenSocketConn([b"AAAA"])
    tts = CartesiaTts(api_key="k", voice_id="v", connect=_connect(conn))

    async def bounded():
        return await asyncio.wait_for(_collect(tts.synthesize(_aiter(["Hi."]))), timeout=5)

    assert asyncio.run(bounded()) == [b"AAAA"]
