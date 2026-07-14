"""Pipeline base class.

A pipeline turns caller input (audio/text) into a spoken/text reply. Both the traditional
(STT→LLM→TTS) and realtime (voice-to-voice) pipelines extend this base so the dispatch core
(VA-21) can treat them uniformly — it satisfies the structural ``dispatch.Pipeline`` protocol.
The traditional pipeline is wired in VA-45 and the realtime pipeline in VA-48.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.dispatch import Architecture
from app.streaming.events import AnySSEEvent
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult


class BasePipeline(ABC):
    """Base class every voice pipeline implements.

    Subclasses set ``architecture`` and implement ``run`` (complete delivery) and ``stream``
    (streaming delivery).
    """

    architecture: Architecture

    @abstractmethod
    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        """Process a full turn and return a complete result."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        """Process a turn and yield SSE events as they are produced."""
        raise NotImplementedError
