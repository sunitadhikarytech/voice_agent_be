"""The four voice endpoints (VA-24..VA-27) + delivery modes (VA-23).

The client selects a pipeline purely by which endpoint it calls — there is no routing field
(VA-20/VA-21). Each endpoint delegates to the shared ``run_for_endpoint`` core, which resolves
the static ``(architecture, delivery)`` for the endpoint and dispatches to the registered
pipeline. Streaming endpoints return Server-Sent Events; ``/complete`` returns one JSON payload.

    /voice/fast      realtime  → SSE      (VA-24)
    /voice/slow      traditional → SSE    (VA-25)
    /voice/complete  traditional → JSON   (VA-26)
    /voice/stream    traditional → SSE    (VA-27)
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.dispatch import Endpoint, run_for_endpoint
from app.streaming.events import AnySSEEvent, to_sse
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult

router = APIRouter()


async def _sse(stream: AsyncIterator[AnySSEEvent]) -> AsyncIterator[str]:
    async for event in stream:
        yield to_sse(event)


async def _stream_response(endpoint: Endpoint, body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    stream = await run_for_endpoint(endpoint, body, registry=request.app.state.pipelines)
    return StreamingResponse(_sse(stream), media_type="text/event-stream")


@router.post("/fast", summary="Fast path (realtime voice-to-voice), streamed as SSE")
async def voice_fast(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    return await _stream_response(Endpoint.FAST, body, request)


@router.post("/slow", summary="Slow path (traditional STT→LLM→TTS), streamed as SSE")
async def voice_slow(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    return await _stream_response(Endpoint.SLOW, body, request)


@router.post(
    "/complete",
    response_model=VoiceTurnResult,
    summary="Traditional path returned as one complete JSON payload",
)
async def voice_complete(body: VoiceTurnRequest, request: Request) -> VoiceTurnResult:
    return await run_for_endpoint(Endpoint.COMPLETE, body, registry=request.app.state.pipelines)


@router.post("/stream", summary="Traditional path as a Server-Sent-Events stream")
async def voice_stream(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    return await _stream_response(Endpoint.STREAM, body, request)
