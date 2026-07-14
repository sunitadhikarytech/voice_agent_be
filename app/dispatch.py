"""Endpoint dispatch core — the shared seam every endpoint calls (VA-21).

There is deliberately NO intent classifier or smart router in this service: the client
chooses a pipeline purely by which endpoint URL it calls. Each of the four endpoints maps
statically to an ``(Architecture, Delivery)`` pair (``ENDPOINT_MAP``), and ``run_turn`` is the
single core every endpoint delegates to. It resolves the pipeline for the architecture from a
registry and either awaits a complete result or returns the stream of SSE events.

Concrete pipelines are registered later (traditional VA-45, realtime VA-48); this module only
defines the seam and is exercised with fakes.
"""
from __future__ import annotations

from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable

from app.streaming.events import AnySSEEvent
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult


class Architecture(str, Enum):
    """Which pipeline handles the turn."""

    TRADITIONAL = "traditional"  # STT -> LLM -> TTS, document-grounded, tool-capable (slow path)
    REALTIME = "realtime"        # voice-to-voice over WebSocket (fast path)


class Delivery(str, Enum):
    """How the result is returned to the caller."""

    COMPLETE = "complete"  # single JSON payload (VoiceTurnResult)
    STREAM = "stream"      # server-sent events


class Endpoint(str, Enum):
    """The four public voice endpoints (VA-24..VA-27)."""

    FAST = "fast"
    SLOW = "slow"
    COMPLETE = "complete"
    STREAM = "stream"


# The only routing input: which URL the client called. No classifier, no request field.
ENDPOINT_MAP: dict[Endpoint, tuple[Architecture, Delivery]] = {
    Endpoint.FAST: (Architecture.REALTIME, Delivery.STREAM),
    Endpoint.SLOW: (Architecture.TRADITIONAL, Delivery.STREAM),
    Endpoint.COMPLETE: (Architecture.TRADITIONAL, Delivery.COMPLETE),
    Endpoint.STREAM: (Architecture.TRADITIONAL, Delivery.STREAM),
}


@runtime_checkable
class Pipeline(Protocol):
    """The contract a pipeline satisfies to be dispatchable."""

    architecture: Architecture

    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        """Process a full turn and return a complete result."""
        ...

    def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        """Process a turn and yield SSE events as they are produced."""
        ...


class DispatchError(RuntimeError):
    """Base class for dispatch errors."""


class PipelineNotRegistered(DispatchError):
    def __init__(self, architecture: Architecture) -> None:
        super().__init__(f"no pipeline registered for architecture '{architecture.value}'")
        self.architecture = architecture


class PipelineRegistry:
    """Holds one pipeline per architecture. Pipelines register themselves at startup."""

    def __init__(self) -> None:
        self._by_architecture: dict[Architecture, Pipeline] = {}

    def register(self, pipeline: Pipeline) -> None:
        self._by_architecture[pipeline.architecture] = pipeline

    def get(self, architecture: Architecture) -> Pipeline:
        try:
            return self._by_architecture[architecture]
        except KeyError as exc:
            raise PipelineNotRegistered(architecture) from exc

    def __contains__(self, architecture: Architecture) -> bool:
        return architecture in self._by_architecture


async def run_turn(
    request: VoiceTurnRequest,
    *,
    architecture: Architecture,
    delivery: Delivery,
    registry: PipelineRegistry,
) -> VoiceTurnResult | AsyncIterator[AnySSEEvent]:
    """Run one voice turn through the pipeline for ``architecture`` in the given ``delivery``
    mode. Returns a :class:`VoiceTurnResult` for ``COMPLETE`` delivery, or an async iterator of
    SSE events for ``STREAM`` delivery. No routing/classification happens here."""
    pipeline = registry.get(architecture)
    if delivery is Delivery.STREAM:
        return pipeline.stream(request)
    return await pipeline.run(request)


async def run_for_endpoint(
    endpoint: Endpoint,
    request: VoiceTurnRequest,
    *,
    registry: PipelineRegistry,
) -> VoiceTurnResult | AsyncIterator[AnySSEEvent]:
    """Convenience wrapper the HTTP endpoints (VA-24..27) use: look up the static
    ``(architecture, delivery)`` for the endpoint and dispatch."""
    architecture, delivery = ENDPOINT_MAP[endpoint]
    return await run_turn(
        request, architecture=architecture, delivery=delivery, registry=registry
    )
