"""Server-Sent-Event contract for streaming voice turns (VA-20).

Defines the event payloads a streaming turn emits, in order:
``transcript.partial`` → ``transcript.final`` → ``answer.delta`` (repeated) →
``audio.chunk`` (repeated) → ``done``. The SSE endpoint (VA-27) produces these; the browser
client (VA-53) consumes them. Each event is a typed model, and ``to_sse`` renders it to the
``event:``/``data:`` wire format.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SSEEventName(str, Enum):
    """Canonical SSE event names (the wire ``event:`` values)."""

    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    ANSWER_DELTA = "answer.delta"
    AUDIO_CHUNK = "audio.chunk"
    DONE = "done"


class TranscriptPartial(BaseModel):
    event: Literal["transcript.partial"] = "transcript.partial"
    text: str


class TranscriptFinal(BaseModel):
    event: Literal["transcript.final"] = "transcript.final"
    text: str


class AnswerDelta(BaseModel):
    event: Literal["answer.delta"] = "answer.delta"
    text: str


class AudioChunk(BaseModel):
    event: Literal["audio.chunk"] = "audio.chunk"
    audio_b64: str = Field(description="Base64-encoded audio bytes for this chunk.")
    seq: int = Field(ge=0, description="Monotonic chunk index within the turn.")


class Done(BaseModel):
    event: Literal["done"] = "done"
    session_id: str | None = None
    latency_ms: dict[str, float] = Field(
        default_factory=dict, description="Per-stage latency for the turn."
    )


# Discriminated on the ``event`` field so a stream can be parsed back into typed events.
AnySSEEvent = Annotated[
    Union[TranscriptPartial, TranscriptFinal, AnswerDelta, AudioChunk, Done],
    Field(discriminator="event"),
]

# Concrete event classes, handy for iteration/validation.
SSE_EVENT_MODELS: tuple[type[BaseModel], ...] = (
    TranscriptPartial,
    TranscriptFinal,
    AnswerDelta,
    AudioChunk,
    Done,
)


def to_sse(event: BaseModel) -> str:
    """Render an event model to the SSE wire format: an ``event:`` line, a JSON ``data:``
    line, and the terminating blank line."""
    name = getattr(event, "event")
    return f"event: {name}\ndata: {event.model_dump_json()}\n\n"


def example_events() -> list[AnySSEEvent]:
    """One example of each event, in emission order — used to publish the contract."""
    return [
        TranscriptPartial(text="what is the ref"),
        TranscriptFinal(text="what is the refund policy?"),
        AnswerDelta(text="Refunds are available "),
        AudioChunk(audio_b64="<base64-opus>", seq=0),
        Done(session_id="sess-123", latency_ms={"stt": 180.0, "llm": 420.0, "tts": 90.0}),
    ]
