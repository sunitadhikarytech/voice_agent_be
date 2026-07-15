"""VA-32 — end-of-turn / turn-taking handling.

``take_turn`` passes STT chunks through for exactly one user turn: it stops at the first
``is_end_of_turn`` (instead of draining a stream that may never end) and always closes the
source. The traditional pipeline joins every final segment into the turn transcript.
"""
from __future__ import annotations

import asyncio

from app.pipelines.traditional import TraditionalPipeline
from app.pipelines.turn_taking import join_segments, take_turn
from app.providers.base import TranscriptChunk
from app.providers.mock import MockLlm, MockTts
from app.streaming.events import TranscriptFinal
from app.streaming.schemas import VoiceTurnRequest


def _chunk(text: str, *, final: bool = False, eot: bool = False) -> TranscriptChunk:
    return TranscriptChunk(text=text, is_final=final, is_end_of_turn=eot)


async def _collect(iterator):
    return [item async for item in iterator]


# --- take_turn ---------------------------------------------------------------------------------

def test_stops_at_end_of_turn_without_pulling_further():
    pulled_past_eot = False

    async def source():
        nonlocal pulled_past_eot
        yield _chunk("hel", final=False)
        yield _chunk("hello", final=True, eot=True)
        pulled_past_eot = True  # a live mic would keep streaming; we must never get here
        yield _chunk("noise after the turn", final=True)

    chunks = asyncio.run(_collect(take_turn(source())))
    assert [c.text for c in chunks] == ["hel", "hello"]
    assert pulled_past_eot is False


def test_stream_ending_without_signal_is_a_complete_turn():
    async def source():
        yield _chunk("finite", final=True)  # no end-of-turn flag anywhere

    chunks = asyncio.run(_collect(take_turn(source())))
    assert [c.text for c in chunks] == ["finite"]


def test_source_closed_on_early_stop():
    closed = False

    async def source():
        try:
            yield _chunk("a", final=True, eot=True)
            yield _chunk("never")
        finally:
            nonlocal closed
            closed = True  # the STT connection is released at the turn boundary

    asyncio.run(_collect(take_turn(source())))
    assert closed is True


def test_source_closed_when_consumer_abandons_the_turn():
    closed = False

    async def source():
        try:
            while True:
                yield _chunk("endless partial")
        finally:
            nonlocal closed
            closed = True

    async def scenario():
        turn = take_turn(source())
        first = await turn.__anext__()
        assert first.text == "endless partial"
        await turn.aclose()  # consumer walks away mid-turn (e.g. client disconnect)

    asyncio.run(scenario())
    assert closed is True


def test_join_segments():
    assert join_segments(["Hello.", " How are you? ", ""]) == "Hello. How are you?"
    assert join_segments([]) == ""


# --- pipeline integration ------------------------------------------------------------------------

class SegmentedStt:
    """An STT stream that emits several final segments before signalling end of turn."""

    name = "segmented"

    def __init__(self, segments: list[str], *, endless_after: bool = False) -> None:
        self._segments = segments
        self._endless_after = endless_after

    async def transcribe(self, audio):
        async for _ in audio:
            pass
        for i, text in enumerate(self._segments):
            last = i == len(self._segments) - 1
            yield TranscriptChunk(text=text, is_final=True, is_end_of_turn=last)
        if self._endless_after:
            while True:  # a live connection: never ends on its own
                yield TranscriptChunk(text="should never be read", is_final=True)


AUDIO = VoiceTurnRequest.model_validate(
    {"input": {"kind": "audio", "audio_b64": "AQID"}}  # base64 of \x01\x02\x03
)


def test_multi_segment_turn_joins_all_finals():
    pipeline = TraditionalPipeline(SegmentedStt(["Hello.", "How are you?"]), MockLlm(), MockTts())
    result = asyncio.run(pipeline.run(AUDIO))
    assert result.transcript == "Hello. How are you?"  # not just the last segment


def test_one_transcript_final_event_per_segment():
    pipeline = TraditionalPipeline(SegmentedStt(["One.", "Two.", "Three."]), MockLlm(), MockTts())
    events = asyncio.run(_collect(pipeline.stream(AUDIO)))
    finals = [e for e in events if isinstance(e, TranscriptFinal)]
    assert [f.text for f in finals] == ["One.", "Two.", "Three."]


def test_pipeline_replies_instead_of_listening_forever():
    stt = SegmentedStt(["the whole question"], endless_after=True)
    pipeline = TraditionalPipeline(stt, MockLlm(), MockTts())
    result = asyncio.run(pipeline.run(AUDIO))  # would hang without end-of-turn handling
    assert result.transcript == "the whole question"
    assert result.answer_text == "mock answer"


def test_text_input_is_a_single_final_turn():
    pipeline = TraditionalPipeline(SegmentedStt([]), MockLlm(), MockTts())
    result = asyncio.run(
        pipeline.run(
            VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": "typed question"}})
        )
    )
    assert result.transcript == "typed question"
