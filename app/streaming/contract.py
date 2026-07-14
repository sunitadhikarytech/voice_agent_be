"""Contract endpoints that publish the shared voice-turn schemas (VA-20).

These are documentation/validation helpers — they do NOT process a turn (there is no unified
router endpoint; the fast/slow/complete/stream endpoints in VA-24..VA-27 are the real entry
points). They exist so the shared request and SSE-event schemas are published in
``/openapi.json`` and so the request contract is verifiable now.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.streaming.events import AnySSEEvent, example_events
from app.streaming.schemas import VoiceTurnRequest


class ContractInfo(BaseModel):
    """The published voice-turn contract: an example request plus one of every SSE event."""

    request_example: VoiceTurnRequest
    sse_events: list[AnySSEEvent]


router = APIRouter()


@router.post(
    "/validate",
    response_model=VoiceTurnRequest,
    summary="Validate a voice-turn request against the shared contract",
)
def validate_request(request: VoiceTurnRequest) -> VoiceTurnRequest:
    """Validate a request body against the shared voice-turn contract and echo it back.

    Returns 422 for an invalid body — including any routing/``architecture`` field, which is
    rejected because the endpoint URL is the only pipeline selector. Does not process a turn.
    """
    return request


@router.get(
    "/schema",
    response_model=ContractInfo,
    summary="Published voice-turn request + SSE event contract",
)
def contract_schema() -> ContractInfo:
    """Return an example request and one example of every SSE event, in emission order."""
    return ContractInfo(
        request_example=VoiceTurnRequest.model_validate(
            {"session_id": "sess-123", "input": {"kind": "text", "text": "hi"}}
        ),
        sse_events=example_events(),
    )
