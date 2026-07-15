"""Assemble the pipeline registry from configuration.

Builds the provider adapters the settings select (VA-30 factory), wraps them in the traditional
(VA-45) and realtime (VA-48) pipelines, and registers both under their architecture so the
endpoints can dispatch by ``run_turn`` (VA-21). Grounding (VA-36/37), tools (VA-38/39), memory
(VA-41) and the session store (VA-40) are wired in here.
"""
from __future__ import annotations

from app.context.loader import DocumentContext
from app.dispatch import PipelineRegistry
from app.observability import EventCounters, LatencyMetrics, UsageMetrics
from app.pipelines.realtime import RealtimePipeline
from app.pipelines.traditional import TraditionalPipeline
from app.providers.factory import get_realtime, make_providers
from app.session import ConversationMemory, SessionStore
from app.tools import ToolRegistry, default_registry


def build_pipeline_registry(
    settings,
    *,
    document: DocumentContext | None = None,
    tools: ToolRegistry | None = None,
    session_store: SessionStore | None = None,
    metrics: LatencyMetrics | None = None,
    usage: UsageMetrics | None = None,
    counters: EventCounters | None = None,
) -> PipelineRegistry:
    """Build and register the traditional + realtime pipelines for ``settings``."""
    session_store = session_store if session_store is not None else SessionStore()
    tools = tools if tools is not None else default_registry()
    memory = ConversationMemory(token_budget=settings.conversation_memory_tokens)

    stt, llm, tts = make_providers(settings)
    realtime = get_realtime(settings.realtime_provider, settings)

    registry = PipelineRegistry()
    registry.register(
        TraditionalPipeline(
            stt, llm, tts,
            session_store=session_store, memory=memory, tools=tools, document=document,
            metrics=metrics, usage=usage, counters=counters,
        )
    )
    registry.register(
        RealtimePipeline(
            realtime, session_store=session_store, metrics=metrics, usage=usage, counters=counters
        )
    )
    return registry
