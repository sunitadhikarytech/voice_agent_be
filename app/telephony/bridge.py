"""Bridge a Twilio Media Stream to the voice pipeline (VA-74).

Turns a phone call into a document-grounded conversation. Caller audio (μ-law 8 kHz) flows
into Deepgram (configured for μ-law), which segments the stream into turns via its end-of-turn
signal (VA-32). Each finished utterance is answered by the grounded LLM (VA-37) and spoken by
the TTS provider; that PCM is resampled + μ-law-encoded and streamed back down the same socket.

The loop is **half-duplex** (listen → answer → listen): while the agent speaks, inbound audio
buffers on the socket and is consumed on the next turn. Full-duplex barge-in is a refinement
(the realtime fast path already has server-side barge-in). Providers are injectable, so the
whole loop runs on mocks with no network.
"""
from __future__ import annotations

import logging

from app.context import ground_llm
from app.pipelines.turn_taking import join_segments
from app.providers.factory import get_llm, get_tts
from app.telephony.audio import pcm16_to_twilio_mulaw
from app.telephony.stream import TwilioMediaStream

logger = logging.getLogger("app.telephony.bridge")

# Cartesia/ElevenLabs emit PCM16 at 24 kHz; Twilio wants μ-law 8 kHz.
TTS_OUTPUT_RATE = 24000
# ~20 ms μ-law frames (8000 Hz × 0.02 s) — Twilio-friendly playback chunks.
FRAME_BYTES = 160
GREETING = "Hello! Ask me anything about the Constitution."


def _build_telephony_stt(settings):
    """STT configured for the telephony wire format. Deepgram decodes μ-law 8 kHz natively
    when told; other providers fall back to their default config (may need transcoding)."""
    if settings.stt_provider == "deepgram":
        from app.providers.deepgram_stt import DeepgramStt

        return DeepgramStt(
            api_key=settings.deepgram_api_key.get_secret_value(),
            model=settings.deepgram_model,
            encoding="mulaw",
            sample_rate=8000,
        )
    from app.providers.factory import get_stt

    return get_stt(settings.stt_provider, settings)


async def run_call(stream: TwilioMediaStream, app_state, *, stt=None, llm=None, tts=None) -> None:
    """Handle one phone call: greet, then answer grounded questions until the caller hangs up.

    Providers default to the app's configured ones; tests inject mocks.
    """
    settings = app_state.settings
    document = getattr(app_state, "document", None)
    stt = stt or _build_telephony_stt(settings)
    llm = llm or get_llm(settings.llm_provider, settings)
    tts = tts or get_tts(settings.tts_provider, settings)
    if document is not None:
        ground_llm(llm, document)  # full-document grounding onto the LLM (VA-37)

    # Wait for Twilio's start (so we have the streamSid) before speaking — outbound media
    # without a streamSid is dropped.
    if not await stream.wait_for_start():
        return
    await _speak(stream, tts, GREETING)

    segments: list[str] = []
    turns = 0
    async for chunk in stt.transcribe(stream.inbound_audio()):
        if chunk.text and chunk.is_final:
            segments.append(chunk.text)
        if not chunk.is_end_of_turn:
            continue
        question = join_segments(segments)
        segments = []
        if not question:
            continue
        turns += 1
        logger.info("telephony question", extra={"call_sid": stream.call_sid, "turn": turns})
        answer = "".join([token async for token in llm.generate(question)])
        if answer:
            await _speak(stream, tts, answer)

    logger.info("telephony call ended", extra={"call_sid": stream.call_sid, "turns": turns})


async def _speak(stream: TwilioMediaStream, tts, text: str) -> None:
    """Synthesize ``text`` and stream it to the caller as μ-law frames, then mark the end."""
    async def _once():
        yield text

    async for pcm in tts.synthesize(_once()):
        mulaw = pcm16_to_twilio_mulaw(pcm, from_rate=TTS_OUTPUT_RATE)
        for i in range(0, len(mulaw), FRAME_BYTES):
            await stream.send_audio(mulaw[i:i + FRAME_BYTES])
    await stream.send_mark("end-of-answer")
