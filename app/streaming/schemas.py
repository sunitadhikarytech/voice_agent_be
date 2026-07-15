"""Shared request schema for a voice turn (VA-20).

Every voice endpoint (fast / slow / complete / stream — VA-24..VA-27) accepts this same
request body. There is deliberately **no** architecture/routing field: the client selects a
pipeline purely by which endpoint URL it calls, so any such field is rejected (``extra`` is
forbidden). The input is a discriminated union of text or webm-opus audio.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class TextInput(BaseModel):
    """A text utterance."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"kind": "text", "text": "What does Article 21 guarantee?"}]},
    )

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, description="User text for this turn.")


class AudioInput(BaseModel):
    """A base64-encoded audio utterance (Opus in a WebM container)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"kind": "audio", "audio_b64": "T2dnUwACAAAA...", "mime": "audio/webm;codecs=opus"}
            ]
        },
    )

    kind: Literal["audio"] = "audio"
    audio_b64: str = Field(min_length=1, description="Base64-encoded audio bytes.")
    mime: Literal["audio/webm;codecs=opus"] = "audio/webm;codecs=opus"


# Discriminated on ``kind`` so the payload is unambiguous and self-describing.
VoiceInput = Annotated[Union[TextInput, AudioInput], Field(discriminator="kind")]


class VoiceTurnRequest(BaseModel):
    """The request body shared by all voice endpoints.

    ``extra="forbid"`` means an unknown field — notably any ``architecture``/``pipeline``/
    ``mode`` routing hint — makes the request invalid (HTTP 422). The endpoint URL is the
    only selector.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"input": {"kind": "text", "text": "What does Article 21 guarantee?"}},
                {
                    "session_id": "sess-8f14e45f",
                    "input": {"kind": "text", "text": "And how do the courts enforce it?"},
                },
                {
                    "input": {
                        "kind": "audio",
                        "audio_b64": "T2dnUwACAAAA...",
                        "mime": "audio/webm;codecs=opus",
                    }
                },
            ]
        },
    )

    session_id: str | None = Field(
        default=None,
        description="Conversation/session id for continuity; omit to start a new session.",
    )
    input: VoiceInput


class VoiceTurnResult(BaseModel):
    """The complete (non-streaming) result of a voice turn.

    Returned by the ``complete`` delivery mode (the `/voice/complete` endpoint, VA-26) and by
    the dispatch core (VA-21). Streaming delivery emits the SSE events in ``app.streaming.events``
    instead.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "session_id": "sess-8f14e45f",
                    "transcript": "What does Article 21 guarantee?",
                    "answer_text": "Article 21 guarantees the protection of life and personal liberty…",
                    "audio_url": None,
                    "tools_called": [],
                    "latency_ms": {"stt_ms": 182.4, "llm_ms": 421.9, "first_audio_ms": 730.2},
                }
            ]
        },
    )

    session_id: str | None = None
    transcript: str = Field(description="Final transcript of the user's utterance.")
    answer_text: str = Field(description="The agent's answer as text.")
    audio_url: str | None = Field(default=None, description="URL/handle for the spoken answer, if any.")
    tools_called: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict, description="Per-stage latency.")
