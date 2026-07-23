"""Groq LLM adapter — a fast, OpenAI-compatible alternate brain (VA-34 alt).

Groq serves open models (Llama et al.) over an OpenAI-compatible streaming chat-completions
API at very low latency and with a generous free tier — a drop-in alternative to the Gemini
adapter when its quota is exhausted, and fast enough to make the traditional voice turn feel
near-instant. Select it with ``LLM_PROVIDER=groq``.

Grounding note: Groq's Llama models have a **128k-token context**, far smaller than the 1M
window the full-document grounding (VA-35/37) assumes. When a ``document_context`` is set it
is inlined into the system prompt but **capped** to a safe size (``max_context_chars``) — a
document larger than that is truncated with a warning, and the model answers from its own
knowledge beyond the cut. For strict whole-document grounding use the Gemini adapter (1M
context). This is *not* a voice-to-voice provider — it fills the LLM role only.

The token stream is injectable (``stream``), so the adapter is fully testable without any
network calls; the default streams from Groq via ``httpx``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Callable, Sequence

logger = logging.getLogger("app.providers.groq")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
# ~128k-token context; keep the inlined doc well under it, leaving room for prompt + answer.
DEFAULT_MAX_CONTEXT_CHARS = 300_000

StreamFn = Callable[..., AsyncIterator[str]]


class GroqLlm:
    """LlmProvider backed by Groq's OpenAI-compatible chat completions."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "You are a helpful voice assistant.",
        base_url: str = DEFAULT_BASE_URL,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        stream: StreamFn | None = None,
        tools: Sequence[Any] | None = None,
        document_context: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_context_chars = max_context_chars
        self.system_prompt = system_prompt
        # Hooks populated by the pipeline / ground_llm; consumed on every generate().
        self.tools = list(tools) if tools else None
        self.document_context = document_context
        self._stream = stream or self._default_stream

    @classmethod
    def from_settings(cls, settings) -> "GroqLlm":
        return cls(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
        )

    def _effective_system(self, override: str | None) -> str:
        """The system message for a turn: the (grounded) system prompt, plus the inlined
        document context when present — capped to the model's context budget."""
        base = override if override is not None else self.system_prompt
        doc = self.document_context
        if not doc:
            return base
        if len(doc) > self._max_context_chars:
            logger.warning(
                "groq: document (%d chars) exceeds the %d-char context budget; truncating — "
                "answers beyond the cut rely on the model's own knowledge",
                len(doc), self._max_context_chars,
            )
            doc = doc[: self._max_context_chars]
        return f"{base}\n\n--- SOURCE DOCUMENT (authoritative) ---\n{doc}"

    def _messages(self, prompt: str, system: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    async def generate(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Stream the answer tokens for ``prompt``."""
        messages = self._messages(prompt, self._effective_system(system))
        async for token in self._stream(messages=messages):
            yield token

    async def _default_stream(self, *, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        # Lazy import so httpx only loads when a real Groq provider streams.
        import httpx

        payload = {"model": self._model, "messages": messages, "stream": True, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
