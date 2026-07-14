"""VA-21 — endpoint dispatch core (no router)."""
import asyncio

import pytest

from app.dispatch import (
    ENDPOINT_MAP,
    Architecture,
    Delivery,
    Endpoint,
    Pipeline,
    PipelineNotRegistered,
    PipelineRegistry,
    run_for_endpoint,
    run_turn,
)
from app.streaming.events import Done, TranscriptFinal
from app.streaming.schemas import VoiceTurnRequest, VoiceTurnResult


class FakePipeline:
    """Records how it was invoked so tests can assert dispatch behaviour."""

    def __init__(self, architecture: Architecture) -> None:
        self.architecture = architecture
        self.calls: list[str] = []

    async def run(self, request: VoiceTurnRequest) -> VoiceTurnResult:
        self.calls.append("run")
        return VoiceTurnResult(
            session_id=request.session_id, transcript="hi", answer_text="hello"
        )

    async def stream(self, request: VoiceTurnRequest):
        self.calls.append("stream")
        yield TranscriptFinal(text="hi")
        yield Done(session_id=request.session_id)


def _req(text: str = "hi") -> VoiceTurnRequest:
    return VoiceTurnRequest.model_validate({"input": {"kind": "text", "text": text}})


def _registry() -> tuple[PipelineRegistry, FakePipeline, FakePipeline]:
    trad = FakePipeline(Architecture.TRADITIONAL)
    rt = FakePipeline(Architecture.REALTIME)
    reg = PipelineRegistry()
    reg.register(trad)
    reg.register(rt)
    return reg, trad, rt


# --- static endpoint mapping (the only routing input) -----------------------------------

def test_endpoint_map_is_complete_and_correct():
    assert ENDPOINT_MAP == {
        Endpoint.FAST: (Architecture.REALTIME, Delivery.STREAM),
        Endpoint.SLOW: (Architecture.TRADITIONAL, Delivery.STREAM),
        Endpoint.COMPLETE: (Architecture.TRADITIONAL, Delivery.COMPLETE),
        Endpoint.STREAM: (Architecture.TRADITIONAL, Delivery.STREAM),
    }
    # every public endpoint is mapped
    assert set(ENDPOINT_MAP) == set(Endpoint)


def test_fake_pipeline_satisfies_protocol():
    assert isinstance(FakePipeline(Architecture.TRADITIONAL), Pipeline)


# --- run_turn ---------------------------------------------------------------------------

def test_complete_delivery_awaits_result():
    reg, trad, _ = _registry()

    result = asyncio.run(
        run_turn(_req(), architecture=Architecture.TRADITIONAL,
                 delivery=Delivery.COMPLETE, registry=reg)
    )
    assert isinstance(result, VoiceTurnResult)
    assert result.answer_text == "hello"
    assert trad.calls == ["run"]


def test_stream_delivery_yields_events():
    reg, trad, _ = _registry()

    async def drive():
        stream = await run_turn(
            _req(), architecture=Architecture.TRADITIONAL,
            delivery=Delivery.STREAM, registry=reg
        )
        return [event async for event in stream]

    events = asyncio.run(drive())
    assert [e.event for e in events] == ["transcript.final", "done"]
    assert trad.calls == ["stream"]


def test_unregistered_architecture_raises():
    reg = PipelineRegistry()
    reg.register(FakePipeline(Architecture.TRADITIONAL))  # no realtime
    with pytest.raises(PipelineNotRegistered) as ei:
        asyncio.run(
            run_turn(_req(), architecture=Architecture.REALTIME,
                     delivery=Delivery.STREAM, registry=reg)
        )
    assert ei.value.architecture is Architecture.REALTIME


# --- run_for_endpoint: every endpoint dispatches to the right pipeline + mode ------------

@pytest.mark.parametrize(
    "endpoint,expected_arch,expected_call",
    [
        (Endpoint.FAST, Architecture.REALTIME, "stream"),
        (Endpoint.SLOW, Architecture.TRADITIONAL, "stream"),
        (Endpoint.COMPLETE, Architecture.TRADITIONAL, "run"),
        (Endpoint.STREAM, Architecture.TRADITIONAL, "stream"),
    ],
)
def test_run_for_endpoint_dispatches_correctly(endpoint, expected_arch, expected_call):
    reg, trad, rt = _registry()
    chosen = rt if expected_arch is Architecture.REALTIME else trad
    other = trad if expected_arch is Architecture.REALTIME else rt

    async def drive():
        out = await run_for_endpoint(endpoint, _req(), registry=reg)
        if expected_call == "stream":
            return [e async for e in out]
        return out

    asyncio.run(drive())
    assert chosen.calls == [expected_call]
    assert other.calls == []  # the other pipeline is never touched


def test_routing_ignores_request_content_no_classifier():
    # Same endpoint always routes the same way regardless of the input payload — proving
    # there is no content-based classifier.
    reg, trad, rt = _registry()

    async def drive(req):
        out = await run_for_endpoint(Endpoint.FAST, req, registry=reg)
        return [e async for e in out]

    asyncio.run(drive(_req("short")))
    asyncio.run(drive(_req("a much longer and more complex utterance about refunds")))
    assert rt.calls == ["stream", "stream"]  # both went to realtime (FAST), never traditional
    assert trad.calls == []
