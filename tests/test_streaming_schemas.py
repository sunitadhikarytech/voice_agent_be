"""VA-20 — shared request schema and SSE event contract."""
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from app.streaming.events import (
    SSE_EVENT_MODELS,
    AnySSEEvent,
    AnswerDelta,
    AudioChunk,
    Done,
    SSEEventName,
    TranscriptPartial,
    to_sse,
)
from app.streaming.schemas import AudioInput, TextInput, VoiceTurnRequest

# --- request schema ---------------------------------------------------------------------

def test_text_input_request_parses():
    req = VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": "hi"}})
    assert isinstance(req.input, TextInput)
    assert req.input.text == "hi"
    assert req.session_id is None


def test_audio_input_request_parses():
    req = VoiceTurnRequest.model_validate(
        {"session_id": "s1", "input": {"kind": "audio", "audio_b64": "AAAA"}}
    )
    assert isinstance(req.input, AudioInput)
    assert req.input.mime == "audio/webm;codecs=opus"
    assert req.session_id == "s1"


def test_no_architecture_or_routing_field_accepted():
    # Any routing hint at the top level is rejected — the endpoint URL is the selector.
    with pytest.raises(ValidationError):
        VoiceTurnRequest.model_validate(
            {"architecture": "realtime", "input": {"kind": "text", "text": "hi"}}
        )


def test_unknown_field_inside_input_rejected():
    with pytest.raises(ValidationError):
        VoiceTurnRequest.model_validate(
            {"input": {"kind": "text", "text": "hi", "speed": "fast"}}
        )


def test_missing_input_rejected():
    with pytest.raises(ValidationError):
        VoiceTurnRequest.model_validate({"session_id": "s1"})


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": ""}})


# --- SSE event contract -----------------------------------------------------------------

def test_event_names_match_enum():
    literal_names = {m.model_fields["event"].default for m in SSE_EVENT_MODELS}
    assert literal_names == {e.value for e in SSEEventName}


def test_to_sse_wire_format():
    wire = to_sse(TranscriptPartial(text="hello"))
    assert wire.startswith("event: transcript.partial\n")
    assert wire.endswith("\n\n")
    data_line = next(ln for ln in wire.splitlines() if ln.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {
        "event": "transcript.partial",
        "text": "hello",
    }


def test_audio_chunk_seq_must_be_non_negative():
    with pytest.raises(ValidationError):
        AudioChunk(audio_b64="AAAA", seq=-1)


def test_any_sse_event_round_trips_by_discriminator():
    adapter = TypeAdapter(AnySSEEvent)
    for original in (AnswerDelta(text="x"), Done(session_id="s1")):
        parsed = adapter.validate_json(original.model_dump_json())
        assert type(parsed) is type(original)
        assert parsed == original
