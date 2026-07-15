"""VA-64 (QA-03) — integration: a full turn through each path, end to end.

Where the contract suite (VA-63) pins the wire *shape*, this exercises *behavior*: a complete
turn drives every stage of the real pipeline (only the network-bound providers are mocked),
sessions persist across separate requests through the shared store (VA-40), and the
observability collectors (latency VA-58, usage VA-59, counters VA-60) reflect the turns that
actually happened.

Each test builds a fresh app so the shared session store and collectors start empty and the
assertions are deterministic.
"""
from __future__ import annotations

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.dispatch import Architecture
from app.main import create_app

PREFIX = "/api/v1"
TRAD = Architecture.TRADITIONAL.value
RT = Architecture.REALTIME.value

_TEXT = {"input": {"kind": "text", "text": "what does article 21 guarantee?"}}
_AUDIO_BYTES = b"\x01\x02\x03\x04"
_AUDIO = {"input": {"kind": "audio", "audio_b64": base64.b64encode(_AUDIO_BYTES).decode()}}


def _build() -> tuple[FastAPI, TestClient]:
    app = create_app(
        Settings(
            _env_file=None,
            stt_provider="mock",
            llm_provider="mock",
            tts_provider="mock",
            realtime_provider="mock",
        )
    )
    return app, TestClient(app)


def _sse_events(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        fields = dict(line.partition(": ")[::2] for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


# --- a full turn completes on every path ------------------------------------------------

def test_complete_full_turn_drives_every_stage():
    _, client = _build()
    body = client.post(PREFIX + "/voice/complete", json=_TEXT).json()
    assert body["transcript"] == "what does article 21 guarantee?"
    assert body["answer_text"] == "mock answer"
    # STT → LLM → TTS all ran, each contributing a latency stage
    assert {"stt_ms", "llm_ms", "first_audio_ms"} <= set(body["latency_ms"])


@pytest.mark.parametrize("path", ["/voice/slow", "/voice/stream"])
def test_traditional_sse_full_turn(path):
    _, client = _build()
    events = _sse_events(client.post(PREFIX + path, json=_TEXT).text)
    kinds = [e for e, _ in events]
    assert "transcript.final" in kinds and kinds[-1] == "done"
    # the answer, reconstructed from the deltas, is the mock answer …
    answer = "".join(d["text"] for e, d in events if e == "answer.delta")
    assert answer == "mock answer"
    # … and the synthesized audio is the answer bytes (MockTts round-trip)
    audio = [d for e, d in events if e == "audio.chunk"]
    assert audio and base64.b64decode(audio[0]["audio_b64"]) == b"mock answer"


def test_fast_realtime_full_turn_echoes_audio():
    _, client = _build()
    events = _sse_events(client.post(PREFIX + "/voice/fast", json=_AUDIO).text)
    assert [e for e, _ in events][-1] == "done"
    audio = [d for e, d in events if e == "audio.chunk"]
    # the realtime mock is voice-to-voice: it echoes input audio back, prefixed
    assert audio and base64.b64decode(audio[0]["audio_b64"]) == b"out:" + _AUDIO_BYTES


def test_audio_input_runs_through_stt_on_the_traditional_path():
    _, client = _build()
    # audio (not text) on /complete must exercise the STT adapter → mock transcript
    body = client.post(PREFIX + "/voice/complete", json=_AUDIO).json()
    assert body["transcript"] == "mock transcript"
    assert body["answer_text"] == "mock answer"


# --- sessions persist across separate requests ------------------------------------------

def test_session_continuity_accumulates_turns():
    app, client = _build()
    sid = "sess-integration"
    for _ in range(2):
        assert (
            client.post(PREFIX + "/voice/complete", json={**_TEXT, "session_id": sid}).status_code
            == 200
        )
    store = app.state.session_store
    assert len(store) == 1  # same session reused across requests, not duplicated
    session = store.get("default", sid)
    assert session is not None
    # two turns → alternating user/agent utterances persisted for continuity
    assert [t.role for t in session.turns] == ["user", "agent", "user", "agent"]
    assert session.turns[1].text == "mock answer"


def test_new_session_per_request_without_id():
    app, client = _build()
    for _ in range(3):
        client.post(PREFIX + "/voice/complete", json=_TEXT)  # no session_id → fresh each time
    assert len(app.state.session_store) == 3


def test_session_is_shared_across_paths():
    app, client = _build()
    sid = "sess-cross"
    client.post(PREFIX + "/voice/slow", json={**_TEXT, "session_id": sid})  # traditional SSE
    client.post(PREFIX + "/voice/complete", json={**_TEXT, "session_id": sid})  # traditional JSON
    assert len(app.state.session_store) == 1
    assert len(app.state.session_store.get("default", sid).turns) == 4


# --- observability reflects the turns that happened -------------------------------------

def test_counters_track_turns_per_path():
    _, client = _build()
    for _ in range(3):
        client.post(PREFIX + "/voice/slow", json=_TEXT)  # traditional
    client.post(PREFIX + "/voice/fast", json=_AUDIO)  # realtime
    counters = client.get(PREFIX + "/counters").json()
    assert counters[TRAD]["turns"] == 3
    assert counters[RT]["turns"] == 1
    assert counters[TRAD]["errors"] == 0
    assert counters[TRAD]["error_rate"] == 0.0


def test_metrics_and_usage_populate_after_a_turn():
    _, client = _build()
    client.post(PREFIX + "/voice/complete", json=_TEXT)
    metrics = client.get(PREFIX + "/metrics").json()
    assert metrics[TRAD]["stt_ms"]["count"] == 1
    assert "llm_ms" in metrics[TRAD] and "first_audio_ms" in metrics[TRAD]
    usage = client.get(PREFIX + "/usage").json()
    assert usage[TRAD]["default"]["turns"] == 1
    assert usage[TRAD]["default"]["tokens"] > 0
