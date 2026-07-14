"""Config-driven provider factory (VA-30).

Maps a provider name to a constructor for each provider type, so selecting a provider is a
config change, not a code change. Concrete adapters register themselves here as they are
added (VA-31/34/43/...); the ``"mock"`` provider is always available for offline tests.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.providers.base import LlmProvider, SttProvider, TtsProvider
from app.providers.mock import MockLlm, MockStt, MockTts

if TYPE_CHECKING:
    from app.config import Settings


class UnknownProvider(KeyError):
    """Raised when a configured provider name has no registered constructor."""


def _build_deepgram_stt(settings: "Settings") -> SttProvider:
    # Lazy import so `websockets` only loads when a Deepgram provider is actually built.
    from app.providers.deepgram_stt import DeepgramStt

    return DeepgramStt.from_settings(settings)


# Constructors receive the app Settings, so adapters can read their model/keys from config.
_STT: dict[str, Callable[["Settings"], SttProvider]] = {
    "mock": lambda _s: MockStt(),
    "deepgram": _build_deepgram_stt,
}
_LLM: dict[str, Callable[["Settings"], LlmProvider]] = {"mock": lambda _s: MockLlm()}
_TTS: dict[str, Callable[["Settings"], TtsProvider]] = {"mock": lambda _s: MockTts()}


def register_stt(name: str, ctor: Callable[["Settings"], SttProvider]) -> None:
    _STT[name] = ctor


def register_llm(name: str, ctor: Callable[["Settings"], LlmProvider]) -> None:
    _LLM[name] = ctor


def register_tts(name: str, ctor: Callable[["Settings"], TtsProvider]) -> None:
    _TTS[name] = ctor


def _build(registry: dict[str, Callable], name: str, kind: str, settings: "Settings"):
    try:
        ctor = registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "(none)"
        raise UnknownProvider(
            f"unknown {kind} provider '{name}'; registered: {available}"
        ) from exc
    return ctor(settings)


def get_stt(name: str, settings: "Settings") -> SttProvider:
    return _build(_STT, name, "STT", settings)


def get_llm(name: str, settings: "Settings") -> LlmProvider:
    return _build(_LLM, name, "LLM", settings)


def get_tts(name: str, settings: "Settings") -> TtsProvider:
    return _build(_TTS, name, "TTS", settings)


def make_providers(settings: "Settings") -> tuple[SttProvider, LlmProvider, TtsProvider]:
    """Build the (STT, LLM, TTS) trio the settings select — the config-driven entry point
    pipelines use."""
    return (
        get_stt(settings.stt_provider, settings),
        get_llm(settings.llm_provider, settings),
        get_tts(settings.tts_provider, settings),
    )
