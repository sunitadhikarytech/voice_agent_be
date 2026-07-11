"""Pipeline interface.

A pipeline turns caller input (audio/text) into a spoken/text reply. Both the traditional
(STT→LLM→TTS) and realtime (voice-to-voice) pipelines implement this contract so the
dispatch core can treat them uniformly. VA-01 defines the contract; the traditional
pipeline is wired in VA-45 and the realtime pipeline in VA-48.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class Pipeline(ABC):
    """Base class every voice pipeline implements."""

    name: str = "pipeline"

    @abstractmethod
    async def run(self, request: Any) -> Any:
        """Process a full turn and return a complete result."""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: Any) -> AsyncIterator[Any]:
        """Process a turn and yield events as they are produced."""
        raise NotImplementedError
