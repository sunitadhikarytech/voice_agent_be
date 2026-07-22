"""Telephony HTTP + WebSocket routes (VA-72).

Mounted at the top level (outside the API prefix) so Twilio's unauthenticated webhook and
media socket are reachable without a bearer token — Twilio authenticates instead via its
request signature (validated here) on the same-origin public URL.

    POST /telephony/voice    → TwiML that connects the call to the media stream
    WS   /telephony/stream   → the bidirectional Media Stream (bridged to the pipeline)
"""
from __future__ import annotations

import logging
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request, Response, WebSocket

from app.telephony.bridge import run_call
from app.telephony.stream import TwilioMediaStream
from app.telephony.twiml import build_stream_twiml, is_valid_signature

logger = logging.getLogger("app.telephony")

router = APIRouter()


def _stream_ws_url(public_base_url: str) -> str:
    """Absolute wss:// URL of the media-stream endpoint, derived from the public origin."""
    base = public_base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/telephony/stream"


@router.post("/telephony/voice", include_in_schema=False)
async def telephony_voice(request: Request) -> Response:
    """Answer an inbound call with TwiML that opens the bidirectional media stream."""
    settings = request.app.state.settings
    if not settings.public_base_url:
        # Without a public origin we cannot hand Twilio a reachable wss URL.
        return Response(status_code=503, content="PUBLIC_BASE_URL is not configured")

    # Twilio POSTs application/x-www-form-urlencoded; parse it directly so we don't pull in
    # python-multipart just for this one webhook.
    raw = (await request.body()).decode("utf-8")
    params = dict(parse_qsl(raw, keep_blank_values=True))

    auth_token = settings.twilio_auth_token.get_secret_value()
    if auth_token:
        url = settings.public_base_url.rstrip("/") + request.url.path
        signature = request.headers.get("X-Twilio-Signature", "")
        if not is_valid_signature(url, params, signature, auth_token):
            logger.warning("rejected telephony webhook: bad Twilio signature")
            return Response(status_code=403, content="invalid Twilio signature")

    twiml = build_stream_twiml(_stream_ws_url(settings.public_base_url))
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/telephony/stream")
async def telephony_stream(websocket: WebSocket) -> None:
    """The Twilio Media Stream for one call — bridged to the voice pipeline."""
    await websocket.accept()
    stream = TwilioMediaStream(websocket)
    try:
        await run_call(stream, websocket.app.state)
    except Exception:
        logger.exception("telephony call failed")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client/disconnect
