"""Audio transcoding for telephony (VA-72).

Twilio Media Streams carry **G.711 μ-law, 8 kHz, mono** in both directions. The voice
pipeline speaks PCM16: Deepgram accepts μ-law 8 kHz natively (so inbound needs no transcode —
we set the STT encoding), but the TTS output is PCM16 at 24 kHz and must be resampled to
8 kHz and μ-law-encoded before it goes back down the phone line.

These are the only two conversions the bridge needs; both are pure functions over ``bytes``,
so they are trivially unit-testable.

``audioop`` (stdlib) provides correct, battle-tested μ-law and resampling. It is deprecated
and removed in Python 3.13 — when moving to 3.13, install the ``audioop-lts`` backport (same
API) or swap in a pure G.711 codec; nothing else here changes.
"""
from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # noqa: E402  (deprecated in 3.13; see module docstring)

TWILIO_SAMPLE_RATE = 8000
TWILIO_ENCODING = "mulaw"
PCM_WIDTH = 2  # 16-bit samples


def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    """Decode μ-law bytes to signed 16-bit little-endian PCM (same sample rate)."""
    return audioop.ulaw2lin(mulaw, PCM_WIDTH)


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Encode signed 16-bit little-endian PCM to μ-law (same sample rate)."""
    return audioop.lin2ulaw(pcm, PCM_WIDTH)


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample mono 16-bit PCM between sample rates."""
    if from_rate == to_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, PCM_WIDTH, 1, from_rate, to_rate, None)
    return converted


def pcm16_to_twilio_mulaw(pcm: bytes, from_rate: int) -> bytes:
    """Full outbound path: PCM16 at ``from_rate`` → μ-law 8 kHz for Twilio playback."""
    return pcm16_to_mulaw(resample_pcm16(pcm, from_rate, TWILIO_SAMPLE_RATE))
