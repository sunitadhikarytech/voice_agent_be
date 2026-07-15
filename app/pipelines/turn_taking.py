"""End-of-turn / turn-taking handling (VA-32).

A conversational agent must know when the user has *finished speaking* — that boundary is
what lets it stop listening and take its own turn. The STT adapters surface the provider's
end-of-turn signal (Deepgram endpointing et al.) as ``TranscriptChunk.is_end_of_turn``;
this module turns that flag into explicit turn-taking:

* :func:`take_turn` passes transcript chunks through **for exactly one user turn** — it
  stops after the first chunk that signals end-of-turn instead of draining the source. On a
  live microphone stream that never ends on its own, this is the difference between
  replying and listening forever. A source that ends *without* the signal (finite uploads,
  adapters without endpointing) is a complete turn too.
* The source generator is **closed deterministically** — on early stop *and* when the
  consumer itself abandons the turn — so a streaming STT connection is released at the turn
  boundary, not whenever the garbage collector notices.

The traditional pipeline (VA-45) consumes this and joins every final segment into the turn
transcript — an utterance like *“Hello. How are you?”* arrives as two finals and must not
collapse to just the last one.
"""
from __future__ import annotations

from typing import AsyncIterator

from app.providers.base import TranscriptChunk


async def take_turn(chunks: AsyncIterator[TranscriptChunk]) -> AsyncIterator[TranscriptChunk]:
    """Yield transcript chunks for exactly one user turn.

    Stops after (and including) the first chunk flagged ``is_end_of_turn``; a source that
    finishes without the flag is treated as a completed turn as well. The source iterator is
    always closed on exit, so the underlying STT stream/connection is released at the turn
    boundary even if this generator itself is abandoned mid-turn.
    """
    try:
        async for chunk in chunks:
            yield chunk
            if chunk.is_end_of_turn:
                return
    finally:
        closer = getattr(chunks, "aclose", None)
        if closer is not None:
            await closer()


def join_segments(segments: list[str]) -> str:
    """Join final transcript segments into the turn transcript."""
    return " ".join(part.strip() for part in segments if part.strip())
