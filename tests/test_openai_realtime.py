"""VA-46 — OpenAI Realtime voice-to-voice adapter (mocked transport; no live calls)."""
import asyncio
import base64
import json

import pytest

from app.config import Settings
from app.providers import factory
from app.providers.base import RealtimeProvider
from app.providers.mock import MockRealtime
from app.providers.openai_realtime import OpenAIRealtime, RealtimeError, parse_audio

S = Settings(_env_file=None)


class FakeConn:
    """Scripted realtime connection.

    - ``send`` genuinely suspends (so background sender tasks really run) and can be
      scripted to fail on the Nth call.
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
        except StopIteration:  # pragma: no cover
            raise ConnectionError("no more connections")

    return _connect


def audio_delta(pcm: bytes) -> str:
    return json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()})


async def _aiter(items):
    for item in items:
        yield item


async def _collect(agen):
    return [item async for item in agen]


def _sent_types(conn: FakeConn) -> list[str]:
    return [json.loads(s)["type"] for s in conn.sent if isinstance(s, str)]


def _sent_appends(conn: FakeConn) -> list[bytes]:
    return [
        base64.b64decode(json.loads(s)["audio"])
        for s in conn.sent
        if isinstance(s, str) and json.loads(s).get("type") == "input_audio_buffer.append"
    ]


def _assert_session_update(frame: str, *, voice: str = "alloy"):
    data = json.loads(frame)
    assert data["type"] == "session.update"
    assert data["session"]["voice"] == voice
    assert data["session"]["turn_detection"]["type"] == "server_vad"


# --- parsing (hardened against untrusted frames) ------------------------------------------

def test_parse_audio_delta_and_noise():
    assert parse_audio(audio_delta(b"xy")) == [b"xy"]
    assert parse_audio(json.dumps({"type": "response.done"})) == []
    assert parse_audio("not json") == []


@pytest.mark.parametrize(
    "frame",
    [
        "[]",                                                # valid JSON, not an object
        "42",                                                # valid JSON scalar
        json.dumps({"type": "response.audio.delta", "delta": ""}),          # empty delta
        json.dumps({"type": "response.audio.delta", "delta": "!!not-b64!!"}),  # bad base64
    ],
)
def test_parse_audio_malformed_frames_are_skipped(frame):
    assert parse_audio(frame) == []


# --- interfaces ---------------------------------------------------------------------------

def test_conforms_to_interface():
    assert isinstance(OpenAIRealtime(api_key="k"), RealtimeProvider)
    assert isinstance(MockRealtime(), RealtimeProvider)


def test_mock_realtime_echoes_audio_and_interrupts():
    mock = MockRealtime()
    out = asyncio.run(_collect(mock.converse(_aiter([b"a", b"b"]))))
    assert out == [b"out:a", b"out:b"]
    asyncio.run(mock.interrupt())
    assert mock.interrupts == 1


# --- streaming ------------------------------------------------------------------------------

def test_audio_in_audio_out_and_session_configured():
    conn = FakeConn(
        [audio_delta(b"AAAA"), audio_delta(b"BBBB"), json.dumps({"type": "response.done"})],
        stop_when=lambda c: len(_sent_appends(c)) == 2,  # let the sender finish first
    )
    rt = OpenAIRealtime(api_key="k", voice="verse", connect=connect_seq(conn))
    out = asyncio.run(_collect(rt.converse(_aiter([b"mic1", b"mic2"]))))

    assert out == [b"AAAA", b"BBBB"]
    _assert_session_update(conn.sent[0], voice="verse")
    assert _sent_appends(conn) == [b"mic1", b"mic2"]
    # server VAD owns turn commits: no manual commit is ever sent
    assert "input_audio_buffer.commit" not in _sent_types(conn)
    assert conn.closed


def test_reconnect_preserves_remaining_audio_and_reconfigures_session():
    """B1 regression: a mid-stream send failure must not kill the shared mic generator —
    the remaining audio flows into the reconnected session."""
    # conn1: receive side blocks forever; 2nd send (the append after session.update) fails.
    conn1 = FakeConn([], block_forever=True, fail_send_at=2)
    # conn2: healthy; ends once it has received the remaining two mic chunks.
    conn2 = FakeConn([audio_delta(b"OUT")], stop_when=lambda c: len(_sent_appends(c)) == 2)
    rt = OpenAIRealtime(
        api_key="k", connect=connect_seq(conn1, conn2), max_reconnects=2, backoff_base=0
    )
    out = asyncio.run(_collect(rt.converse(_aiter([b"m1", b"m2", b"m3"]))))

    assert out == [b"OUT"]
    # sessions are per-socket: config re-sent on the recovered connection
    _assert_session_update(conn1.sent[0])
    _assert_session_update(conn2.sent[0])
    # m1 was lost in-flight with the failed socket; m2/m3 survived to the new session
    assert _sent_appends(conn2) == [b"m2", b"m3"]
    assert conn1.closed and conn2.closed


def test_receive_side_drop_reconnects():
    dropped = FakeConn([audio_delta(b"AAAA")], drop_after=1)
    recovered = FakeConn([audio_delta(b"BBBB")])
    rt = OpenAIRealtime(
        api_key="k", connect=connect_seq(dropped, recovered), max_reconnects=2, backoff_base=0
    )
    out = asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))
    assert out == [b"AAAA", b"BBBB"]
    assert dropped.closed and recovered.closed


def test_initial_connect_failures_retry_then_succeed():
    conn = FakeConn([audio_delta(b"OK")])
    calls = {"n": 0}

    async def flaky_connect():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("connect refused")
        return conn

    rt = OpenAIRealtime(api_key="k", connect=flaky_connect, max_reconnects=2, backoff_base=0)
    out = asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))
    assert out == [b"OK"]
    assert calls["n"] == 3


def test_gives_up_after_max_reconnects():
    async def always_fails():
        raise ConnectionError("refused")

    rt = OpenAIRealtime(api_key="k", connect=always_fails, max_reconnects=2, backoff_base=0)
    with pytest.raises(RealtimeError):
        asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))


def test_non_recoverable_error_propagates_unwrapped():
    async def broken_connect():
        raise ValueError("programming error")

    rt = OpenAIRealtime(api_key="k", connect=broken_connect, max_reconnects=2, backoff_base=0)
    with pytest.raises(ValueError):
        asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))


def test_retry_budget_resets_after_healthy_output():
    """S2: two separate single-drop incidents must both recover under max_reconnects=1."""
    c1 = FakeConn([audio_delta(b"A")], drop_after=1)
    c2 = FakeConn([audio_delta(b"B")], drop_after=1)
    c3 = FakeConn([audio_delta(b"C")])
    rt = OpenAIRealtime(
        api_key="k", connect=connect_seq(c1, c2, c3), max_reconnects=1, backoff_base=0
    )
    out = asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))
    assert out == [b"A", b"B", b"C"]


# --- barge-in interrupt ---------------------------------------------------------------------

def test_interrupt_during_active_session():
    """Interrupt fired while converse() is genuinely streaming reaches the live socket."""
    rt_holder: dict = {}

    async def run():
        conn = FakeConn(
            [audio_delta(b"AAAA")],
            stop_when=lambda c: "response.cancel" in _sent_types(c),
        )
        rt = OpenAIRealtime(api_key="k", connect=connect_seq(conn))
        rt_holder["rt"] = rt
        out = []

        async def consume():
            async for audio in rt.converse(_aiter([b"mic"])):
                out.append(audio)

        consumer = asyncio.create_task(consume())
        while not out:  # wait until the session is demonstrably live
            await asyncio.sleep(0)
        await rt.interrupt()
        await consumer
        return conn, out

    conn, out = asyncio.run(run())
    assert out == [b"AAAA"]
    types = _sent_types(conn)
    assert "response.cancel" in types and "input_audio_buffer.clear" in types


def test_interrupt_is_a_noop_without_active_session():
    rt = OpenAIRealtime(api_key="k")  # no active connection
    asyncio.run(rt.interrupt())  # must not raise


# --- security -------------------------------------------------------------------------------

def test_api_key_never_appears_in_frames_or_repr():
    secret = "sk-super-secret-key"
    conn = FakeConn([audio_delta(b"A")])
    rt = OpenAIRealtime(api_key=secret, connect=connect_seq(conn))
    asyncio.run(_collect(rt.converse(_aiter([b"mic"]))))
    assert all(secret not in s for s in conn.sent if isinstance(s, str))
    assert secret not in repr(rt)


def test_non_wss_url_is_rejected():
    with pytest.raises(ValueError):
        OpenAIRealtime(api_key="k", url="ws://insecure.example/v1/realtime")


# --- config + factory ------------------------------------------------------------------------

def test_from_settings_reads_config():
    settings = Settings(
        _env_file=None, openai_realtime_model="gpt-4o-realtime-preview", openai_voice="verse"
    )
    rt = OpenAIRealtime.from_settings(settings)
    assert (rt.name, rt._model, rt._voice) == ("openai", "gpt-4o-realtime-preview", "verse")


def test_factory_resolves_realtime_providers():
    assert isinstance(factory.get_realtime("mock", S), MockRealtime)
    assert isinstance(factory.get_realtime("openai", S), OpenAIRealtime)
