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

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, description="User text for this turn.")


class AudioInput(BaseModel):
    """A base64-encoded audio utterance (Opus in a WebM container)."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        description="Conversation/session id for continuity; omit to start a new session.",
    )
    input: VoiceInput
