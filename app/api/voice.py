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
from app.streaming.events import AnySSEEvent, example_events, to_sse
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult

router = APIRouter()

# One canonical rendered stream (VA-29): the same example events the contract endpoint
# publishes, in SSE wire form, shown for every streaming endpoint in Swagger.
_SSE_EXAMPLE = "".join(to_sse(event) for event in example_events())

# OpenAPI documentation for the streaming endpoints: FastAPI cannot infer the body of a
# StreamingResponse, so the event-stream content is declared explicitly.
_STREAM_RESPONSES: dict[int | str, dict] = {
    200: {
        "description": (
            "A Server-Sent-Events stream. Each frame is an `event:` name plus a JSON `data:` "
            "payload (schemas under *contract*), terminating with a single `done` event."
        ),
        "content": {
            "text/event-stream": {
                "schema": {"type": "string", "format": "event-stream"},
                "example": _SSE_EXAMPLE,
            }
        },
    }
}


async def _sse(stream: AsyncIterator[AnySSEEvent]) -> AsyncIterator[str]:
    async for event in stream:
        yield to_sse(event)


async def _stream_response(endpoint: Endpoint, body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    stream = await run_for_endpoint(endpoint, body, registry=request.app.state.pipelines)
    return StreamingResponse(_sse(stream), media_type="text/event-stream")


@router.post(
    "/fast",
    summary="Fast path (realtime voice-to-voice), streamed as SSE",
    responses=_STREAM_RESPONSES,
)
async def voice_fast(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    """Low-latency voice-to-voice turn over the **realtime** pipeline.

    Send **audio** input; the reply streams back as `audio.chunk` events (PCM16 @ 24 kHz,
    base64) and terminates with `done`. There is no separate transcript/answer text on this
    path. Choosing this URL *is* choosing the realtime pipeline — there is no routing field.
    """
    return await _stream_response(Endpoint.FAST, body, request)


@router.post(
    "/slow",
    summary="Slow path (traditional STT→LLM→TTS), streamed as SSE",
    responses=_STREAM_RESPONSES,
)
async def voice_slow(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    """Document-grounded, tool-capable turn over the **traditional** pipeline.

    Emits `transcript.partial`/`transcript.final` (audio input is transcribed; text input
    echoes as a final transcript), then `answer.delta` tokens, then `audio.chunk` speech,
    then `done` with per-stage latency. Grounded in the full source document when configured.
    """
    return await _stream_response(Endpoint.SLOW, body, request)


@router.post(
    "/complete",
    response_model=VoiceTurnResult,
    summary="Traditional path returned as one complete JSON payload",
    response_description="The finished turn: transcript, answer text, and per-stage latency.",
)
async def voice_complete(body: VoiceTurnRequest, request: Request) -> VoiceTurnResult:
    """The same traditional turn as `/voice/slow`, but buffered server-side and returned as
    one JSON object — for callers that don't consume SSE (batch jobs, simple integrations)."""
    return await run_for_endpoint(Endpoint.COMPLETE, body, registry=request.app.state.pipelines)


@router.post(
    "/stream",
    summary="Traditional path as a Server-Sent-Events stream",
    responses=_STREAM_RESPONSES,
)
async def voice_stream(body: VoiceTurnRequest, request: Request) -> StreamingResponse:
    """Explicit streaming-delivery alias of the traditional pipeline (same event grammar as
    `/voice/slow`); kept as its own URL so delivery mode is always endpoint-addressed."""
    return await _stream_response(Endpoint.STREAM, body, request)
