"""Gemini LLM adapter — the traditional-path brain (VA-34, VA-36).

Drives a Gemini Flash-tier model: takes the transcript, applies the agent system prompt, and
streams answer tokens. Exposes ``tools`` for function calling (VA-38).

VA-36 adds full-document grounding with **prompt caching**: when a ``document_context`` is
set, the document (plus the system prompt) is uploaded once as Gemini cached content and every
turn references that cache instead of re-sending the large context — so repeat turns are
billed at the cached-input rate rather than re-processing the whole document. If caching is
disabled (or a per-call system override is given), the document is inlined into the system
instruction as a fallback.

The token stream and the cache creation are both injectable, so the adapter is fully testable
without any network calls; the defaults use ``google-genai``.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

# A token-stream function: given prompt/system/tools/cache_ref, yield answer text chunks.
StreamFn = Callable[..., AsyncIterator[str]]
# Creates cached content from the document text and returns its handle/name.
CreateCacheFn = Callable[[str], Awaitable[str]]


class GeminiLlm:
    """LlmProvider backed by Gemini (Flash-tier), with cached full-document grounding."""

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
        enable_caching: bool = True,
        create_cache: CreateCacheFn | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.system_prompt = system_prompt
        # Hooks populated by later tickets; consumed on every generate().
        self.tools = list(tools) if tools else None
        self.document_context = document_context
        self._enable_caching = enable_caching
        self._stream = stream or self._default_stream
        self._create_cache = create_cache or self._default_create_cache
        # The document is cached once, then reused across turns.
        self._cache_ref: str | None = None
        self._cache_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings) -> "GeminiLlm":
        return cls(
            api_key=settings.google_api_key.get_secret_value(),
            model=settings.gemini_model,
            system_prompt=settings.gemini_system_prompt,
            enable_caching=settings.gemini_enable_prompt_caching,
        )

    def _inline_system(self, override: str | None) -> str:
        """System instruction with the document inlined (fallback when not caching)."""
        base = override or self.system_prompt
        if self.document_context:
            return (
                f"{base}\n\n# Source document (answer strictly from the text below)\n"
                f"{self.document_context}"
            )
        return base

    async def _ensure_cache(self) -> str:
        """Create the cached document content once (idempotent under concurrency)."""
        if self._cache_ref is None:
            async with self._cache_lock:
                if self._cache_ref is None:
                    assert self.document_context is not None
                    self._cache_ref = await self._create_cache(self.document_context)
        return self._cache_ref

    async def _default_create_cache(self, document: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        cache = await client.aio.caches.create(
            model=self._model,
            config=types.CreateCachedContentConfig(
                system_instruction=self.system_prompt, contents=[document]
            ),
        )
        return cache.name

    async def _default_stream(
        self,
        *,
        prompt: str,
        system: str,
        tools: Sequence[Any] | None,
        cache_ref: str | None = None,
    ) -> AsyncIterator[str]:
        # Lazy import so google-genai only loads when a real Gemini provider streams.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        config = types.GenerateContentConfig(
            # When cached, the system instruction + document already live in the cache.
            system_instruction=None if cache_ref else system,
            tools=tools or None,
            cached_content=cache_ref,
        )
        response = await client.aio.models.generate_content_stream(
            model=self._model, contents=prompt, config=config
        )
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def generate(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Stream the answer tokens for ``prompt``.

        With a document and caching enabled (and no per-call system override), the document is
        served from cached content; otherwise it is inlined into the system instruction.
        """
        cache_ref: str | None = None
        if system is None and self.document_context and self._enable_caching:
            cache_ref = await self._ensure_cache()

        effective_system = self.system_prompt if cache_ref else self._inline_system(system)
        stream = self._stream(
            prompt=prompt, system=effective_system, tools=self.tools, cache_ref=cache_ref
        )
        async for token in stream:
            yield token
