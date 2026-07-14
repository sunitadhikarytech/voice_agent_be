"""Streaming layer.

Owns the shared voice-turn request schema and the server-sent-event contract
(transcript.partial/final, answer.delta, audio.chunk, done). Defined in VA-20 and served by
the SSE endpoint in VA-27.
"""
from app.streaming.events import (
    AnswerDelta,
    AnySSEEvent,
    AudioChunk,
    Done,
    SSEEventName,
    TranscriptFinal,
    TranscriptPartial,
    to_sse,
)
from app.streaming.schemas import AudioInput, TextInput, VoiceInput, VoiceTurnRequest

__all__ = [
    "VoiceTurnRequest",
    "VoiceInput",
    "TextInput",
    "AudioInput",
    "SSEEventName",
    "TranscriptPartial",
    "TranscriptFinal",
    "AnswerDelta",
    "AudioChunk",
    "Done",
    "AnySSEEvent",
    "to_sse",
]
