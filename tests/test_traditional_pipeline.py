"""VA-45 — traditional pipeline (STT → LLM → TTS end to end)."""
import asyncio
import base64

from app.context import GROUNDING_INSTRUCTIONS
from app.context.loader import DocumentContext
from app.dispatch import Architecture, Delivery, PipelineRegistry, run_turn
from app.pipelines.traditional import TraditionalPipeline
from app.providers.mock import MockStt, MockTts
from app.session import SessionStore, TurnState, TurnStateMachine
from app.streaming.events import AnswerDelta, AudioChunk, Done, TranscriptFinal, TranscriptPartial
from app.streaming.schemas import VoiceTurnRequest
from app.tools import default_registry


class FakeLlm:
    """LLM that records the prompt it receives and streams canned tokens."""

    name = "fake"

    def __init__(self, tokens=("mock ", "answer")):
        self.system_prompt = "You are helpful."
        self.document_context = None
        self.tools = None
        self.prompts: list[str] = []
        self._tokens = tokens

    async def generate(self, prompt: str, *, system=None):
        self.prompts.append(prompt)
        for token in self._tokens:
            yield token


def _text_request(text="what is article 21?", session_id=None) -> VoiceTurnRequest:
    return VoiceTurnRequest.model_validate(
        {"session_id": session_id, "input": {"kind": "text", "text": text}}
    )


def _audio_request(session_id=None) -> VoiceTurnRequest:
    b64 = base64.b64encode(b"\x00\x01\x02").decode()
    return VoiceTurnRequest.model_validate(
        {"session_id": session_id, "input": {"kind": "audio", "audio_b64": b64}}
    )


def _pipeline(**kwargs) -> tuple[TraditionalPipeline, FakeLlm]:
    llm = kwargs.pop("llm", None) or FakeLlm()
    pipe = TraditionalPipeline(MockStt(), llm, MockTts(), **kwargs)
    return pipe, llm


def _run_stream(pipe, request):
    async def drive():
        return [e async for e in pipe.stream(request)]

    return asyncio.run(drive())


# --- end-to-end streaming ---------------------------------------------------------------

def test_text_turn_streams_events_in_order():
    pipe, _ = _pipeline()
    events = _run_stream(pipe, _text_request("what is article 21?"))

    kinds = [e.event for e in events]
    assert kinds == ["transcript.final", "answer.delta", "answer.delta", "audio.chunk", "done"]
    assert events[0].text == "what is article 21?"
    assert "".join(e.text for e in events if isinstance(e, AnswerDelta)) == "mock answer"
    assert isinstance(events[-1], Done)


def test_audio_turn_runs_stt_and_emits_transcript_events():
    pipe, _ = _pipeline()
    events = _run_stream(pipe, _audio_request())
    assert any(isinstance(e, TranscriptPartial) for e in events)
    final = [e for e in events if isinstance(e, TranscriptFinal)]
    assert final and final[0].text == "mock transcript"
    assert any(isinstance(e, AudioChunk) for e in events)


def test_audio_chunks_are_sequential():
    pipe, _ = _pipeline()
    events = _run_stream(pipe, _text_request())
    seqs = [e.seq for e in events if isinstance(e, AudioChunk)]
    assert seqs == list(range(len(seqs)))


def test_done_carries_session_and_latency():
    pipe, _ = _pipeline()
    done = _run_stream(pipe, _text_request())[-1]
    assert isinstance(done, Done) and done.session_id
    for key in ("stt_ms", "llm_ms", "first_audio_ms"):
        assert key in done.latency_ms and done.latency_ms[key] >= 0


# --- complete delivery ------------------------------------------------------------------

def test_run_returns_complete_result():
    pipe, _ = _pipeline()
    result = asyncio.run(pipe.run(_text_request("hello")))
    assert result.transcript == "hello"
    assert result.answer_text == "mock answer"
    assert "stt_ms" in result.latency_ms


# --- session memory ---------------------------------------------------------------------

def test_injected_session_store_is_used():
    # regression: an empty injected store is falsy (SessionStore.__len__) and must not be
    # silently replaced — a new turn should get an id from the injected store's factory.
    store = SessionStore(id_factory=lambda: "sess-generated")
    pipe, _ = _pipeline(session_store=store)
    done = _run_stream(pipe, _text_request())[-1]  # no session_id -> new session
    assert done.session_id == "sess-generated"


def test_conversation_memory_carries_prior_turns_into_later_prompts():
    store = SessionStore(id_factory=lambda: "sess-1")
    pipe, llm = _pipeline(session_store=store)
    _run_stream(pipe, _text_request("first question", session_id="sess-1"))
    _run_stream(pipe, _text_request("second question", session_id="sess-1"))

    # the 2nd turn's prompt includes the 1st turn's user + agent lines
    second_prompt = llm.prompts[1]
    assert "first question" in second_prompt
    assert "mock answer" in second_prompt
    assert "second question" in second_prompt


# --- state machine ----------------------------------------------------------------------

def test_turn_drives_the_state_machine():
    machine = TurnStateMachine()
    pipe, _ = _pipeline(state_factory=lambda: machine)
    _run_stream(pipe, _text_request())
    assert [(t.frm, t.to) for t in machine.history] == [
        (TurnState.IDLE, TurnState.LISTENING),
        (TurnState.LISTENING, TurnState.THINKING),
        (TurnState.THINKING, TurnState.SPEAKING),
        (TurnState.SPEAKING, TurnState.IDLE),
    ]


# --- grounding + tools wiring -----------------------------------------------------------

def test_grounding_and_tools_are_wired_onto_the_llm():
    llm = FakeLlm()
    doc = DocumentContext(text="DOCTEXT", path="/data/c.pdf", char_count=7, estimated_tokens=2)
    TraditionalPipeline(MockStt(), llm, MockTts(), document=doc, tools=default_registry())
    assert llm.document_context == "DOCTEXT"
    assert GROUNDING_INSTRUCTIONS in llm.system_prompt
    assert any(d["name"] == "book_appointment" for d in llm.tools)


# --- dispatch integration ---------------------------------------------------------------

def test_pipeline_dispatches_via_run_turn():
    pipe, _ = _pipeline()
    registry = PipelineRegistry()
    registry.register(pipe)

    async def drive():
        stream = await run_turn(
            _text_request(), architecture=Architecture.TRADITIONAL,
            delivery=Delivery.STREAM, registry=registry,
        )
        return [e async for e in stream]

    events = asyncio.run(drive())
    assert isinstance(events[-1], Done)
