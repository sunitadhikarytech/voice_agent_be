"""VA-72 — Twilio inbound webhook + Media Streams transport (mocked; no live calls)."""
from __future__ import annotations

import base64
import json

from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.deepgram_stt import DeepgramStt
from app.telephony.audio import (
    TWILIO_SAMPLE_RATE,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    pcm16_to_twilio_mulaw,
    resample_pcm16,
)
from app.telephony.stream import TwilioMediaStream
from app.telephony.twiml import build_stream_twiml, expected_signature, is_valid_signature

PUBLIC = "https://call.example.com"
TOKEN = "twilio-auth-token-abcdef"


# --- audio transcoding -------------------------------------------------------------------------

def test_mulaw_pcm_roundtrip_is_close():
    # μ-law is lossy, but a round-trip must stay close for mid-range samples
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in (0, 1000, -1000, 8000, -8000))
    back = mulaw_to_pcm16(pcm16_to_mulaw(pcm))
    assert len(back) == len(pcm)
    orig = [int.from_bytes(pcm[i:i+2], "little", signed=True) for i in range(0, len(pcm), 2)]
    got = [int.from_bytes(back[i:i+2], "little", signed=True) for i in range(0, len(back), 2)]
    for o, g in zip(orig, got):
        assert abs(o - g) <= max(200, abs(o) * 0.1)


def test_mulaw_is_one_byte_per_sample():
    pcm = b"\x00\x00" * 100          # 100 PCM16 samples = 200 bytes
    assert len(pcm16_to_mulaw(pcm)) == 100  # μ-law is 1 byte/sample


def test_resample_changes_length_by_ratio():
    pcm_24k = b"\x00\x00" * 2400     # 2400 samples @ 24 kHz = 100 ms
    pcm_8k = resample_pcm16(pcm_24k, 24000, 8000)
    assert abs(len(pcm_8k) // 2 - 800) <= 4  # ~800 samples @ 8 kHz


def test_resample_noop_when_rates_match():
    pcm = b"\x01\x02" * 10
    assert resample_pcm16(pcm, 8000, 8000) == pcm


def test_outbound_path_yields_8k_mulaw():
    pcm_24k = b"\x00\x00" * 2400     # 100 ms @ 24 kHz
    mulaw = pcm16_to_twilio_mulaw(pcm_24k, from_rate=24000)
    assert abs(len(mulaw) - TWILIO_SAMPLE_RATE // 10) <= 4  # ~800 μ-law bytes = 100 ms @ 8 kHz


# --- TwiML + signature -------------------------------------------------------------------------

def test_twiml_connects_a_stream():
    xml = build_stream_twiml("wss://call.example.com/telephony/stream")
    assert xml.startswith("<?xml")
    assert "<Connect>" in xml and "<Stream" in xml
    assert 'url="wss://call.example.com/telephony/stream"' in xml


def test_signature_validation_roundtrip():
    url = f"{PUBLIC}/telephony/voice"
    params = {"CallSid": "CA123", "From": "+15551112222", "To": "+15553334444"}
    sig = expected_signature(url, params, TOKEN)
    assert is_valid_signature(url, params, sig, TOKEN)
    assert not is_valid_signature(url, params, sig, "wrong-token")
    assert not is_valid_signature(url, {**params, "From": "+1999"}, sig, TOKEN)
    assert not is_valid_signature(url, params, "", TOKEN)


# --- Media Streams transport -------------------------------------------------------------------

class FakeWS:
    """Scripts inbound Twilio frames; records outbound frames. Disconnects when drained."""

    def __init__(self, inbound: list[dict]):
        self._inbound = [json.dumps(m) for m in inbound]
        self.sent: list[dict] = []

    async def receive_text(self) -> str:
        if self._inbound:
            return self._inbound.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))


def _media(payload: bytes) -> dict:
    return {"event": "media", "media": {"payload": base64.b64encode(payload).decode()}}


async def _collect(agen):
    return [x async for x in agen]


def test_transport_captures_stream_sid_and_yields_caller_audio():
    ws = FakeWS([
        {"event": "connected"},
        {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}},
        _media(b"\x01\x02"),
        _media(b"\x03\x04"),
        {"event": "stop"},
        _media(b"\xff"),  # after stop — must not be yielded
    ])
    stream = TwilioMediaStream(ws)
    import asyncio

    audio = asyncio.run(_collect(stream.inbound_audio()))
    assert audio == [b"\x01\x02", b"\x03\x04"]
    assert stream.stream_sid == "MZ1" and stream.call_sid == "CA1"


def test_transport_ends_cleanly_on_disconnect():
    import asyncio

    ws = FakeWS([{"event": "start", "start": {"streamSid": "MZ2"}}, _media(b"\x01")])
    # no stop frame -> the disconnect ends iteration
    assert asyncio.run(_collect(TwilioMediaStream(ws).inbound_audio())) == [b"\x01"]


def test_transport_send_audio_and_control_frames():
    import asyncio

    ws = FakeWS([])
    stream = TwilioMediaStream(ws)
    stream.stream_sid = "MZ9"

    async def scenario():
        await stream.send_audio(b"\x10\x11")
        await stream.send_mark("done")
        await stream.clear()
        await stream.send_audio(b"")  # empty -> no frame

    asyncio.run(scenario())
    events = [f["event"] for f in ws.sent]
    assert events == ["media", "mark", "clear"]  # empty send produced nothing
    assert base64.b64decode(ws.sent[0]["media"]["payload"]) == b"\x10\x11"
    assert all(f["streamSid"] == "MZ9" for f in ws.sent)


# --- Deepgram telephony encoding ---------------------------------------------------------------

def test_deepgram_adds_mulaw_encoding_for_telephony():
    q = DeepgramStt("k", encoding="mulaw", sample_rate=8000)._connect_query()
    assert "encoding=mulaw" in q and "sample_rate=8000" in q


def test_deepgram_default_has_no_encoding():
    assert "encoding=" not in DeepgramStt("k")._connect_query()


# --- the voice webhook -------------------------------------------------------------------------

def _client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(
        _env_file=None, telephony_enabled=True, public_base_url=PUBLIC,
        stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock",
        **overrides,
    )))


def test_voice_webhook_returns_stream_twiml():
    resp = _client().post("/telephony/voice", data={"CallSid": "CA1", "From": "+1555"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert 'url="wss://call.example.com/telephony/stream"' in resp.text


def test_voice_webhook_503_without_public_url():
    client = TestClient(create_app(Settings(
        _env_file=None, telephony_enabled=True, public_base_url="",
        stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock",
    )))
    assert client.post("/telephony/voice", data={"CallSid": "CA1"}).status_code == 503


def test_voice_webhook_rejects_bad_signature_when_token_set():
    client = _client(twilio_auth_token=TOKEN)
    # no/!valid X-Twilio-Signature header -> 403
    assert client.post("/telephony/voice", data={"CallSid": "CA1"}).status_code == 403


def test_voice_webhook_accepts_valid_signature():
    client = _client(twilio_auth_token=TOKEN)
    params = {"CallSid": "CA1", "From": "+1555"}
    sig = expected_signature(f"{PUBLIC}/telephony/voice", params, TOKEN)
    resp = client.post("/telephony/voice", data=params, headers={"X-Twilio-Signature": sig})
    assert resp.status_code == 200


def test_telephony_routes_absent_when_disabled():
    client = TestClient(create_app(Settings(
        _env_file=None, telephony_enabled=False,
        stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock",
    )))
    assert client.post("/telephony/voice", data={"CallSid": "CA1"}).status_code == 404


def test_stream_ws_drains_a_call():
    # the WS route + transport + placeholder bridge complete a start/media/stop lifecycle
    client = _client()
    with client.websocket_connect("/telephony/stream") as ws:
        ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}}))
        ws.send_text(json.dumps(_media(b"\x01\x02")))
        ws.send_text(json.dumps({"event": "stop"}))
    # exiting the context closes cleanly; no exception == the lifecycle handled ok
