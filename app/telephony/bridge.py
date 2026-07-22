"""Bridge a Twilio Media Stream to the voice pipeline (VA-74 fills this in).

VA-72 ships the transport and this seam; VA-74 replaces :func:`run_call` with the real
conversational loop (caller μ-law → Deepgram STT → grounded Gemini → Cartesia → μ-law back).
For now it drains inbound audio so the socket lifecycle is correct end to end.
"""
from __future__ import annotations

import logging

from app.telephony.stream import TwilioMediaStream

logger = logging.getLogger("app.telephony.bridge")


async def run_call(stream: TwilioMediaStream, app_state) -> None:
    """Handle one phone call over ``stream``. Placeholder: drains caller audio until the
    call ends (VA-74 turns this into STT → LLM → TTS turns)."""
    frames = 0
    async for _chunk in stream.inbound_audio():
        frames += 1
    logger.info(
        "telephony call ended", extra={"call_sid": stream.call_sid, "media_frames": frames}
    )
