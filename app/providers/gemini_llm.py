"""Gemini LLM adapter — the traditional-path brain (VA-34).

Drives a Gemini Flash-tier model: takes the transcript, applies the agent system prompt, and
streams answer tokens. It exposes the hooks later tickets wire in — ``tools`` for function
calling (VA-38) and ``document_context`` for full-document grounding (VA-36) — without
changing the ``LlmProvider`` interface.

The token stream is injectable (``stream``) so the adapter is fully testable without any
network calls; the default streams from ``google-genai``.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Sequence

# A token-stream function: given prompt/system/tools, yield answer text chunks.
# (Implements the app.providers.base.LlmProvider structural interface.)
StreamFn = Callable[..., AsyncIterator[str]]


class GeminiLlm:
    """LlmProvider backed by Gemini (Flash-tier)."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "You are a helpful voice assistant.",
        stream: StreamFn | None = None,
        tools: Sequence[Any] | None = None,
        document_context: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.system_prompt = system_prompt
        # Hooks populated by later tickets; consumed on every generate().
        self.tools = list(tools) if tools else None
        self.document_context = document_context
        self._stream = stream or self._default_stream

    @classmethod
    def from_settings(cls, settings) -> "GeminiLlm":
        return cls(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.gemini_model,
            system_prompt=settings.gemini_system_prompt,
        )

    def _effective_system(self, override: str | None) -> str:
        """Compose the system instruction: caller override (or the configured prompt), plus
        the full-document context when set (VA-36 replaces this with cached context)."""
        base = override or self.system_prompt
        if self.document_context:
            return (
                f"{base}\n\n# Source document (answer strictly from the text below)\n"
                f"{self.document_context}"
            )
        return base

    async def _default_stream(
        self, *, prompt: str, system: str, tools: Sequence[Any] | None
    ) -> AsyncIterator[str]:
        # Lazy import so google-genai only loads when a real Gemini provider streams.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        config = types.GenerateContentConfig(system_instruction=system, tools=tools or None)
        response = await client.aio.models.generate_content_stream(
            model=self._model, contents=prompt, config=config
        )
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def generate(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Stream the answer tokens for ``prompt``. Applies the effective system prompt
        (with document context, if any) and the configured tools."""
        stream = self._stream(
            prompt=prompt, system=self._effective_system(system), tools=self.tools
        )
        async for token in stream:
            yield token
