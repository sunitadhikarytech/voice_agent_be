"""VA-31 — Deepgram streaming STT adapter (mocked transport; no live calls)."""
import asyncio
import json

import pytest

from app.config import Settings
from app.providers.base import SttProvider
from app.providers.deepgram_stt import DeepgramStt, SttConnectionError, parse_message


# --- fake transport ---------------------------------------------------------------------

class FakeConn:
    """Scripted Deepgram connection.

    - ``send`` genuinely suspends (so background sender tasks really run) and can be
      scripted to fail on the Nth call via ``fail_send_at``.
    - after the scripted messages are exhausted the receive side either ends
      (StopAsyncIteration), blocks until ``stop_when(self)`` turns true, or blocks forever.
    - ``drop_after`` simulates a receive-side connection drop after N messages.
    """

    def __init__(self, messages, *, drop_after=None, fail_send_at=None, stop_when=None,
                 block_forever=False):
        self._messages = list(messages)
        self._drop_after = drop_after
        self._fail_send_at = fail_send_at
        self._stop_when = stop_when
        self._block_forever = block_forever
        self.sent: list = []
        self.closed = False
        self._i = 0
        self._sends = 0

    async def send(self, data):
        await asyncio.sleep(0)  # a real socket send suspends; critical for task interleaving
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
            if self._block_forever:
                await asyncio.Event().wait()  # never set: receive side hangs
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
        except StopIteration:  # pragma: no cover - exhausted only on over-reconnect tests
            raise ConnectionError("no more connections")

    return _connect


def results(text, *, is_final=False, speech_final=False):
    return json.dumps(
        {
            "type": "Results",
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text}]},
        }
    )


async def _aiter(items):
    for item in items:
        yield item


async def _collect(agen):
    return [item async for item in agen]


def _sent_audio(conn: FakeConn) -> list[bytes]:
    """The raw audio chunks sent on a connection (excludes the CloseStream control frame)."""
    return [s for s in conn.sent if isinstance(s, (bytes, bytearray))]


# --- message parsing --------------------------------------------------------------------

def test_parse_partial_final_and_end_of_turn():
    (partial,) = parse_message(results("hi", is_final=False))
    assert (partial.text, partial.is_final, partial.is_end_of_turn) == ("hi", False, False)

    (final,) = parse_message(results("hi there", is_final=True, speech_final=True))
    assert (final.is_final, final.is_end_of_turn) == (True, True)


def test_parse_utterance_end_and_noise():
    (eot,) = parse_message(json.dumps({"type": "UtteranceEnd"}))
    assert eot.is_end_of_turn and eot.is_final
    assert parse_message(results("")) == []  # empty interim
    assert parse_message(json.dumps({"type": "Metadata"})) == []
    assert parse_message("not json") == []


# --- streaming --------------------------------------------------------------------------

def test_conforms_to_interface():
    assert isinstance(DeepgramStt(api_key="k"), SttProvider)


def test_happy_path_yields_partial_then_final_and_sends_close():
    conn = FakeConn(
        [results("hello", is_final=False), results("hello world", is_final=True, speech_final=True)],
        # keep the receive side open until the sender has flushed the audio + CloseStream
        stop_when=lambda c: any(isinstance(s, str) and "CloseStream" in s for s in c.sent),
    )
    stt = DeepgramStt(api_key="k", connect=connect_seq(conn))
    chunks = asyncio.run(_collect(stt.transcribe(_aiter([b"a", b"b"]))))

    assert [(c.text, c.is_final, c.is_end_of_turn) for c in chunks] == [
        ("hello", False, False),
        ("hello world", True, True),
    ]
    assert conn.closed
    assert _sent_audio(conn) == [b"a", b"b"]
    assert any(isinstance(s, str) and "CloseStream" in s for s in conn.sent)


def test_reconnects_mid_stream_and_continues():
    dropped = FakeConn([results("hel", is_final=False)], drop_after=1)  # 1 msg then drop
    recovered = FakeConn([results("hello world", is_final=True, speech_final=True)])
    stt = DeepgramStt(
        api_key="k", connect=connect_seq(dropped, recovered), max_reconnects=2, backoff_base=0
    )
    chunks = asyncio.run(_collect(stt.transcribe(_aiter([b"a", b"b", b"c"]))))

    assert [c.text for c in chunks] == ["hel", "hello world"]
    assert dropped.closed and recovered.closed  # both connections cleaned up


def test_reconnect_preserves_remaining_audio_after_send_failure():
    """VA-31 follow-up regression: a mid-stream send failure must not finalize the shared
    audio source. Only the in-flight chunk is lost; the remaining audio flows into the
    reconnected session (previously the sender's cancellation killed the shared generator,
    silently dropping the rest of the audio)."""
    # conn1: receive side blocks forever; the first audio send fails mid-stream.
    conn1 = FakeConn([], block_forever=True, fail_send_at=1)
    # conn2: healthy; ends once it has received the remaining two audio chunks.
    conn2 = FakeConn(
        [results("m2 m3", is_final=True, speech_final=True)],
        stop_when=lambda c: len(_sent_audio(c)) == 2,
    )
    stt = DeepgramStt(
        api_key="k", connect=connect_seq(conn1, conn2), max_reconnects=2, backoff_base=0
    )
    chunks = asyncio.run(_collect(stt.transcribe(_aiter([b"m1", b"m2", b"m3"]))))

    assert [c.text for c in chunks] == ["m2 m3"]
    # m1 was lost in-flight on the failed socket; m2/m3 survived into the new session
    assert _sent_audio(conn1) == []
    assert _sent_audio(conn2) == [b"m2", b"m3"]
    assert conn1.closed and conn2.closed


def test_gives_up_after_max_reconnects():
    conns = [FakeConn([], drop_after=0) for _ in range(5)]  # every connection drops immediately
    stt = DeepgramStt(
        api_key="k", connect=connect_seq(*conns), max_reconnects=2, backoff_base=0
    )
    with pytest.raises(SttConnectionError):
        asyncio.run(_collect(stt.transcribe(_aiter([b"a"]))))


def test_from_settings_reads_config():
    settings = Settings(_env_file=None, deepgram_model="nova-3")
    stt = DeepgramStt.from_settings(settings)
    assert stt.name == "deepgram" and stt._model == "nova-3"
