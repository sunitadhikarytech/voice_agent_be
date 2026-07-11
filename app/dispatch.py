"""Endpoint dispatch core — the shared seam every endpoint calls.

There is deliberately NO intent classifier or smart router in this service: the client
chooses a pipeline purely by which endpoint URL it calls. `run_turn` is the single core
each endpoint delegates to, mapping (architecture, delivery) to the right pipeline and
delivery mode.

VA-01 only defines the seam and its vocabulary. The real implementation lands in VA-21
(dispatch core) and the delivery modes in VA-23.
"""
from __future__ import annotations

from enum import Enum


class Architecture(str, Enum):
    """Which pipeline handles the turn."""

    TRADITIONAL = "traditional"  # STT -> LLM -> TTS, document-grounded, tool-capable (slow path)
    REALTIME = "realtime"        # voice-to-voice over WebSocket (fast path)


class Delivery(str, Enum):
    """How the result is returned to the caller."""

    COMPLETE = "complete"  # single JSON payload
    STREAM = "stream"      # server-sent events / streamed audio


async def run_turn(architecture: Architecture, delivery: Delivery, **kwargs):
    """Run one voice turn through the selected pipeline and delivery mode.

    Implemented in VA-21. Present here so the module boundary and the no-router contract
    are established from day one.
    """
    raise NotImplementedError("run_turn is implemented in VA-21 (endpoint dispatch core).")
