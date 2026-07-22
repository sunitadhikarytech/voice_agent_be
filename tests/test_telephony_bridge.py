"""VA-74 — phone-audio bridge: a full call answered over the Media Stream (mocked, no network)."""
from __future__ import annotations

import asyncio
import base64
import json

from fastapi import WebSocketDisconnect

from app.config import Settings
from app.providers.base import TranscriptChunk
from app.providers.mock import MockLlm
from app.telephony.audio import mulaw_to_pcm16
from app.telephony.bridge import _build_telephony_stt, run_call
from app.telephony.stream import TwilioMediaStream


class FakeWS:
    """Scripts inbound Twilio frames; records outbound. Disconnects when drained."""

    def __init__(self, inbound):
        self._inbound = [json.dumps(m) for m in inbound]
        self.sent = []

    async def receive_text(self):
        if self._inbound:
            return self._inbound.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, data):
        self.sent.append(json.loads(data))


def _media(payload=b"\x7f\x7f"):
    return {"event": "media", "media": {"payload": base64.b64encode(payload).decode()}}


class State:
    def __init__(self, **overrides):
        self.settings = Settings(
            _env_file=None, telephony_enabled=True,
            stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock",
            **overrides,
        )
        self.document = None


class ScriptedStt:
    """Drains caller audio, then emits scripted transcript turns (each ends the turn)."""

    def __init__(self, utterances):
        self._utterances = utterances

    async def transcribe(self, audio):
        async for _ in audio:
            pass
        for text in self._utterances:
            yield TranscriptChunk(text=text, is_final=False)
            yield TranscriptChunk(text=text, is_final=True, is_end_of_turn=True)


class SilentStt:
    async def transcribe(self, audio):
        async for _ in audio:
            pass
        return
        yield  # pragma: no cover — marker


class PcmTts:
    """Emits 100 ms of even-length PCM16 @ 24 kHz per text chunk (real-shaped audio)."""

    def __init__(self):
        self.spoken = []

    async def synthesize(self, text):
        async for t in text:
            self.spoken.append(t)
            yield b"\x11\x22" * 2400


def _run(ws, state, **providers):
    asyncio.run(run_call(TwilioMediaStream(ws), state, **providers))


def _events(ws):
    return [f["event"] for f in ws.sent]


def _mulaw_bytes(ws):
    return b"".join(base64.b64decode(f["media"]["payload"]) for f in ws.sent if f["event"] == "media")


# --- a full answered turn ----------------------------------------------------------------------

def test_call_greets_then_answers_a_question():
    ws = FakeWS([
        {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}},
        _media(), _media(),
        {"event": "stop"},
    ])
    tts = PcmTts()
    _run(ws, State(), stt=ScriptedStt(["what is article 21"]), llm=MockLlm(), tts=tts)

    # greeting + one answer were both synthesized
    assert tts.spoken[0].startswith("Hello")
    assert "mock answer" in tts.spoken
    # two spoken segments => two end-of-answer marks; audio streamed as media frames
    assert _events(ws).count("mark") == 2
    assert _events(ws).count("media") > 0
    assert len(_mulaw_bytes(ws)) > 0


def test_media_frames_are_twilio_sized():
    ws = FakeWS([{"event": "start", "start": {"streamSid": "MZ1"}}, _media(), {"event": "stop"}])
    _run(ws, State(), stt=ScriptedStt(["q"]), llm=MockLlm(), tts=PcmTts())
    media = [f for f in ws.sent if f["event"] == "media"]
    for f in media:
        assert len(base64.b64decode(f["media"]["payload"])) <= 160  # ~20 ms μ-law frames
    assert all(f["streamSid"] == "MZ1" for f in ws.sent)


def test_outbound_audio_is_valid_mulaw():
    ws = FakeWS([{"event": "start", "start": {"streamSid": "MZ1"}}, _media(), {"event": "stop"}])
    _run(ws, State(), stt=ScriptedStt(["q"]), llm=MockLlm(), tts=PcmTts())
    pcm = mulaw_to_pcm16(_mulaw_bytes(ws))  # decodes without error
    assert len(pcm) == len(_mulaw_bytes(ws)) * 2


# --- multi-turn + edge cases -------------------------------------------------------------------

def test_multiple_turns_each_get_an_answer():
    ws = FakeWS([
        {"event": "start", "start": {"streamSid": "MZ1"}},
        _media(), {"event": "stop"},
    ])
    tts = PcmTts()
    _run(ws, State(), stt=ScriptedStt(["first question", "second question"]), llm=MockLlm(), tts=tts)
    # greeting + 2 answers
    assert _events(ws).count("mark") == 3
    assert tts.spoken.count("mock answer") == 2


def test_silent_call_only_greets():
    ws = FakeWS([{"event": "start", "start": {"streamSid": "MZ1"}}, _media(), {"event": "stop"}])
    tts = PcmTts()
    _run(ws, State(), stt=SilentStt(), llm=MockLlm(), tts=tts)
    assert tts.spoken == ["Hello! Ask me anything about the Constitution."]
    assert _events(ws).count("mark") == 1  # greeting only, no answer


def test_disconnect_without_stop_ends_the_call():
    # no explicit stop frame — the WS disconnect must end the call cleanly
    ws = FakeWS([{"event": "start", "start": {"streamSid": "MZ1"}}, _media()])
    _run(ws, State(), stt=ScriptedStt(["q"]), llm=MockLlm(), tts=PcmTts())
    assert _events(ws).count("mark") == 2  # completed greeting + answer before disconnect


# --- telephony STT wiring ----------------------------------------------------------------------

def test_telephony_stt_uses_mulaw_for_deepgram():
    from app.providers.deepgram_stt import DeepgramStt

    stt = _build_telephony_stt(Settings(_env_file=None, stt_provider="deepgram"))
    assert isinstance(stt, DeepgramStt)
    q = stt._connect_query()
    assert "encoding=mulaw" in q and "sample_rate=8000" in q


def test_telephony_stt_falls_back_for_other_providers():
    from app.providers.mock import MockStt

    stt = _build_telephony_stt(Settings(_env_file=None, stt_provider="mock"))
    assert isinstance(stt, MockStt)
