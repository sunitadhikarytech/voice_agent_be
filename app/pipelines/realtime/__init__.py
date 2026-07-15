"""Realtime pipeline: voice-to-voice over WebSocket (fast path).

Wraps the realtime adapter (VA-46) as a ``Pipeline`` (VA-48) so the fast endpoint (VA-24) can
dispatch to it via ``run_turn``.
"""
from app.pipelines.realtime.pipeline import RealtimePipeline

__all__ = ["RealtimePipeline"]
