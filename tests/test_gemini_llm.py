"""VA-34 — Gemini LLM adapter (injected token stream; no live calls)."""
import asyncio

from app.config import Settings
from app.providers.base import LlmProvider
from app.providers.gemini_llm import GeminiLlm


class FakeStream:
    """Records how it was called and yields canned tokens."""

    def __init__(self, tokens=("Hello", ", ", "world")):
        self._tokens = tokens
        self.calls: list[dict] = []

    async def __call__(self, *, prompt, system, tools):
        self.calls.append({"prompt": prompt, "system": system, "tools": tools})
        for token in self._tokens:
            yield token


async def _collect(agen):
    return [item async for item in agen]


def _llm(**kwargs) -> tuple[GeminiLlm, FakeStream]:
    fake = FakeStream(kwargs.pop("tokens", ("Hello", ", ", "world")))
    return GeminiLlm(api_key="k", stream=fake, **kwargs), fake


def test_conforms_to_interface():
    llm, _ = _llm()
    assert isinstance(llm, LlmProvider)


def test_generate_streams_tokens():
    llm, _ = _llm()
    tokens = asyncio.run(_collect(llm.generate("hi")))
    assert "".join(tokens) == "Hello, world"


def test_default_system_prompt_used():
    llm, fake = _llm(system_prompt="SYS-DEFAULT")
    asyncio.run(_collect(llm.generate("hi")))
    assert fake.calls[0]["system"] == "SYS-DEFAULT"


def test_system_prompt_override_changes_behaviour():
    llm, fake = _llm(system_prompt="SYS-DEFAULT")
    asyncio.run(_collect(llm.generate("hi", system="SYS-OVERRIDE")))
    assert fake.calls[0]["system"] == "SYS-OVERRIDE"


def test_document_context_hook_is_injected_into_system():
    llm, fake = _llm(system_prompt="SYS")
    llm.document_context = "THE-CONSTITUTION-TEXT"
    asyncio.run(_collect(llm.generate("hi")))
    system = fake.calls[0]["system"]
    assert "SYS" in system and "THE-CONSTITUTION-TEXT" in system


def test_tools_hook_is_passed_through():
    llm, fake = _llm()
    llm.tools = [{"name": "book_appointment"}]
    asyncio.run(_collect(llm.generate("hi")))
    assert fake.calls[0]["tools"] == [{"name": "book_appointment"}]


def test_from_settings_reads_config():
    settings = Settings(_env_file=None, gemini_model="gemini-2.0-flash",
                        gemini_system_prompt="from-config")
    llm = GeminiLlm.from_settings(settings)
    assert llm.name == "gemini"
    assert llm._model == "gemini-2.0-flash"
    assert llm.system_prompt == "from-config"
