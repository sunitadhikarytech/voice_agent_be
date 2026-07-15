"""Realtime → traditional fallback (VA-49).

The fast path depends on one external realtime connection; when it fails the user should
get a (slower) answer, not an error. This wrapper implements that degradation:

* A failure **before the primary has delivered anything** re-runs the whole turn through
  the traditional pipeline — same request, since the traditional path accepts the same
  audio input via its STT stage — and streams *its* events instead. The client sees a
  normal (slow-path) turn.
* A failure **after events have been delivered** propagates: replaying the turn would
  duplicate audio the client already played, so mid-stream errors stay errors.

Every fallback is recorded on the VA-60 counters (``fallback_rate`` per path) and logged.
The realtime pipeline's own turn/error counting still happens inside it, so metering stays
honest: a fallback turn counts as a realtime turn + error *and* a traditional turn *and*
one realtime fallback.

Enabled by default; ``REALTIME_FALLBACK_ENABLED=false`` restores fail-fast behaviour.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from app.dispatch import Architecture, Pipeline
from app.observability import EventCounters
from app.pipelines.base import BasePipeline
from app.streaming.events import AnySSEEvent
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult

logger = logging.getLogger("app.pipelines.fallback")


class RealtimeWithFallback(BasePipeline):
    """The realtime pipeline, degrading to the traditional one on early failure."""

    architecture = Architecture.REALTIME

    def __init__(
        self,
        primary: Pipeline,
        fallback: Pipeline,
        *,
        counters: EventCounters | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._counters = counters

    async def stream(self, request: VoiceTurnRequest) -> AsyncIterator[AnySSEEvent]:
        primary = self._primary.stream(request)
        try:
            first = await primary.__anext__()
        except StopAsyncIteration:
            return  # ended without any event: nothing delivered, nothing to recover
        except ValueError:
            raise  # a caller error (e.g. text input on the fast path) — not an outage
        except Exception as exc:
            # Nothing was delivered yet — the turn can be replayed safely on the slow path.
            async for event in self._fall_back(request, exc):
                yield event
            return

        yield first
        # After the first delivered event a replay would duplicate audio — propagate.
        async for event in primary:
            yield event

    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        try:
            return await self._primary.run(request)
        except ValueError:
            raise  # caller error — not an outage
        except Exception as exc:
            self._record(exc)
            return await self._fallback.run(request)

    def _record(self, exc: Exception) -> None:
        if self._counters is not None:
            self._counters.fallback(self.architecture.value)
        logger.warning(
            "realtime failed before first event; falling back to traditional",
            extra={"error_type": type(exc).__name__},
        )

    async def _fall_back(
        self, request: VoiceTurnRequest, exc: Exception
    ) -> AsyncIterator[AnySSEEvent]:
        self._record(exc)
        async for event in self._fallback.stream(request):
            yield event
