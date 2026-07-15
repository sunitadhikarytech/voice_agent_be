"""VA-63 (QA-02) — contract tests for the four voice endpoints.

Pins the *wire contract* every client depends on, independent of the pipeline internals:

* the request body all four endpoints accept, and the fields they reject (HTTP 422);
* the SSE event grammar for the streaming endpoints — event names, framing, typed
  payloads, ``audio.chunk`` sequencing, ordering, and a single terminating ``done``;
* the JSON result schema for ``/complete``;
* discoverability of all four paths (and the published SSE contract) via OpenAPI.

Mock providers make every turn run end to end with no network access, so these assertions
exercise the real request→dispatch→pipeline→serialization path, not a stub of it.
"""
from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app.config import Settings
from app.main import create_app
from app.streaming.events import AnySSEEvent, SSEEventName
from app.streaming.schemas import VoiceTurnResult

PREFIX = "/api/v1"
STREAMING = ["/voice/fast", "/voice/slow", "/voice/stream"]
TRADITIONAL_SSE = ["/voice/slow", "/voice/stream"]
ALL_ENDPOINTS = STREAMING + ["/voice/complete"]

# Validates any emitted frame back into the discriminated-union event models.
_EVENTS = TypeAdapter(AnySSEEvent)
_CANONICAL = {e.value for e in SSEEventName}


def _client() -> TestClient:
    return TestClient(
        create_app(
            Settings(
                _env_file=None,
                stt_provider="mock",
                llm_provider="mock",
                tts_provider="mock",
                realtime_provider="mock",
            )
        )
    )


client = _client()

_TEXT = {"input": {"kind": "text", "text": "what does article 21 guarantee?"}}
_AUDIO = {"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01\x02\x03").decode()}}


def _body_for(path: str) -> dict:
    """The realtime fast path consumes audio; the traditional paths take text."""
    return _AUDIO if path.endswith("/fast") else _TEXT


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event_name, data_dict), ...]``, asserting the wire framing:
    each frame is one ``event:`` line, one ``data:`` line of JSON, and a blank separator."""
    frames: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            assert ": " in line, f"malformed SSE line: {line!r}"
            key, _, value = line.partition(": ")
            fields[key] = value
        assert set(fields) == {"event", "data"}, f"unexpected SSE fields: {sorted(fields)}"
        frames.append((fields["event"], json.loads(fields["data"])))
    return frames


# --- SSE framing + typed payloads -------------------------------------------------------

@pytest.mark.parametrize("path", STREAMING)
def test_streaming_endpoint_is_event_stream(path):
    resp = client.post(PREFIX + path, json=_body_for(path))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize("path", STREAMING)
def test_every_event_is_named_and_typed(path):
    frames = _parse_sse(client.post(PREFIX + path, json=_body_for(path)).text)
    assert frames, "stream produced no events"
    for name, data in frames:
        assert name in _CANONICAL, f"undocumented SSE event: {name}"
        model = _EVENTS.validate_python(data)  # round-trips through the typed contract
        assert model.event == name


@pytest.mark.parametrize("path", STREAMING)
def test_stream_terminates_with_exactly_one_done(path):
    names = [n for n, _ in _parse_sse(client.post(PREFIX + path, json=_body_for(path)).text)]
    assert names[-1] == "done"
    assert names.count("done") == 1


@pytest.mark.parametrize("path", STREAMING)
def test_audio_chunks_are_sequential_and_decodable(path):
    frames = _parse_sse(client.post(PREFIX + path, json=_body_for(path)).text)
    chunks = [d for n, d in frames if n == "audio.chunk"]
    assert chunks, "no audio chunks emitted"
    assert [c["seq"] for c in chunks] == list(range(len(chunks)))  # monotonic from 0
    for c in chunks:
        assert c["audio_b64"], "empty audio chunk"
        base64.b64decode(c["audio_b64"])  # valid base64, no exception


@pytest.mark.parametrize("path", TRADITIONAL_SSE)
def test_traditional_event_ordering(path):
    names = [n for n, _ in _parse_sse(client.post(PREFIX + path, json=_TEXT).text)]
    # transcript.final → answer.delta → audio.chunk → done
    assert names.index("transcript.final") < names.index("answer.delta")
    assert names.index("answer.delta") < names.index("audio.chunk")
    assert names.index("audio.chunk") < names.index("done")


@pytest.mark.parametrize("path", STREAMING)
def test_done_echoes_session_id_and_carries_latency(path):
    body = {**_body_for(path), "session_id": "sess-contract"}
    frames = _parse_sse(client.post(PREFIX + path, json=body).text)
    done = next(d for n, d in frames if n == "done")
    assert done["session_id"] == "sess-contract"  # resolve() echoes a supplied id
    assert isinstance(done["latency_ms"], dict) and done["latency_ms"]


# --- /complete JSON contract ------------------------------------------------------------

def test_complete_returns_json_result_matching_schema():
    resp = client.post(PREFIX + "/voice/complete", json=_TEXT)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    result = VoiceTurnResult.model_validate(resp.json())  # validates the full shape
    assert result.transcript == "what does article 21 guarantee?"
    assert result.answer_text == "mock answer"
    assert isinstance(result.latency_ms, dict) and result.latency_ms
    assert result.tools_called == []


# --- request contract enforcement (all four endpoints) ----------------------------------

@pytest.mark.parametrize("path", ALL_ENDPOINTS)
def test_missing_input_is_422_problem(path):
    resp = client.post(PREFIX + path, json={"session_id": "s1"})
    assert resp.status_code == 422
    assert resp.json()["status"] == 422  # RFC7807-style problem shape (VA-28)


@pytest.mark.parametrize("path", ALL_ENDPOINTS)
def test_routing_field_is_rejected(path):
    # extra="forbid": the endpoint URL is the only pipeline selector, never a body field.
    resp = client.post(
        PREFIX + path,
        json={"architecture": "realtime", "input": {"kind": "text", "text": "hi"}},
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("path", ALL_ENDPOINTS)
def test_empty_text_is_rejected(path):
    resp = client.post(PREFIX + path, json={"input": {"kind": "text", "text": ""}})
    assert resp.status_code == 422  # text has min_length=1


@pytest.mark.parametrize("path", ALL_ENDPOINTS)
def test_get_is_method_not_allowed(path):
    assert client.get(PREFIX + path).status_code == 405


# --- discoverability --------------------------------------------------------------------

def test_all_four_paths_published_in_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    for p in ALL_ENDPOINTS:
        assert PREFIX + p in paths
        assert "post" in paths[PREFIX + p]


def test_complete_declares_result_model_in_openapi():
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "VoiceTurnResult" in schemas


def test_published_sse_contract_matches_emitted_events():
    # the names advertised at /contract/schema are exactly the canonical event set…
    published = [e["event"] for e in client.get(PREFIX + "/contract/schema").json()["sse_events"]]
    assert set(published) == _CANONICAL
    # …and the traditional stream only ever emits names from that set.
    emitted = {n for n, _ in _parse_sse(client.post(PREFIX + "/voice/slow", json=_TEXT).text)}
    assert emitted <= set(published)
