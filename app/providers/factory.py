"""Config-driven provider factory (VA-30).

Maps a provider name to a constructor for each provider type, so selecting a provider is a
config change, not a code change. Concrete adapters register themselves here as they are
added (VA-31/34/43/...); the ``"mock"`` provider is always available for offline tests.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.providers.base import LlmProvider, RealtimeProvider, SttProvider, TtsProvider
from app.providers.mock import MockLlm, MockRealtime, MockStt, MockTts

if TYPE_CHECKING:
    from app.config import Settings


class UnknownProvider(KeyError):
    """Raised when a configured provider name has no registered constructor."""


def _build_deepgram_stt(settings: "Settings") -> SttProvider:
    # Lazy import so `websockets` only loads when a Deepgram provider is actually built.
    from app.providers.deepgram_stt import DeepgramStt

    return DeepgramStt.from_settings(settings)


def _build_elevenlabs_stt(settings: "Settings") -> SttProvider:
    # Lazy import so `websockets` only loads when an ElevenLabs provider is actually built.
    from app.providers.elevenlabs_stt import ElevenLabsStt

    return ElevenLabsStt.from_settings(settings)


def _build_gemini_llm(settings: "Settings") -> LlmProvider:
    # Lazy import so `google-genai` only loads when a Gemini provider is actually built.
    from app.providers.gemini_llm import GeminiLlm

    return GeminiLlm.from_settings(settings)


def _build_cartesia_tts(settings: "Settings") -> TtsProvider:
    # Lazy import so `websockets` only loads when a Cartesia provider is actually built.
    from app.providers.cartesia_tts import CartesiaTts

    return CartesiaTts.from_settings(settings)


def _build_elevenlabs_tts(settings: "Settings") -> TtsProvider:
    # Lazy import so `websockets` only loads when an ElevenLabs provider is actually built.
    from app.providers.elevenlabs_tts import ElevenLabsTts

    return ElevenLabsTts.from_settings(settings)


def _build_openai_realtime(settings: "Settings") -> RealtimeProvider:
    # Lazy import so `websockets` only loads when a realtime provider is actually built.
    from app.providers.openai_realtime import OpenAIRealtime

    return OpenAIRealtime.from_settings(settings)


def _build_gemini_live(settings: "Settings") -> RealtimeProvider:
    # Lazy import so `google-genai` only loads when a live provider is actually built.
    from app.providers.gemini_live import GeminiLive

    return GeminiLive.from_settings(settings)


def _build_grok_realtime(settings: "Settings") -> RealtimeProvider:
    # Lazy import so `websockets` only loads when a realtime provider is actually built.
    from app.providers.grok_realtime import GrokRealtime

    return GrokRealtime.from_settings(settings)


# Constructors receive the app Settings, so adapters can read their model/keys from config.
_STT: dict[str, Callable[["Settings"], SttProvider]] = {
    "mock": lambda _s: MockStt(),
    "deepgram": _build_deepgram_stt,
    "elevenlabs": _build_elevenlabs_stt,  # alternate STT (VA-33)
}
_LLM: dict[str, Callable[["Settings"], LlmProvider]] = {
    "mock": lambda _s: MockLlm(),
    "gemini": _build_gemini_llm,
}
_TTS: dict[str, Callable[["Settings"], TtsProvider]] = {
    "mock": lambda _s: MockTts(),
    "cartesia": _build_cartesia_tts,
    "elevenlabs": _build_elevenlabs_tts,  # alternate TTS (VA-44)
}
_RT: dict[str, Callable[["Settings"], RealtimeProvider]] = {
    "mock": lambda _s: MockRealtime(),
    "openai": _build_openai_realtime,
    "gemini-live": _build_gemini_live,  # alternate realtime (VA-50)
    "grok": _build_grok_realtime,  # alternate realtime (VA-50)
}


def register_stt(name: str, ctor: Callable[["Settings"], SttProvider]) -> None:
    _STT[name] = ctor


def register_llm(name: str, ctor: Callable[["Settings"], LlmProvider]) -> None:
    _LLM[name] = ctor


def register_tts(name: str, ctor: Callable[["Settings"], TtsProvider]) -> None:
    _TTS[name] = ctor


def register_realtime(name: str, ctor: Callable[["Settings"], RealtimeProvider]) -> None:
    _RT[name] = ctor


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


def get_realtime(name: str, settings: "Settings") -> RealtimeProvider:
    return _build(_RT, name, "realtime", settings)


def make_providers(settings: "Settings") -> tuple[SttProvider, LlmProvider, TtsProvider]:
    """Build the (STT, LLM, TTS) trio the settings select — the config-driven entry point
    pipelines use."""
    return (
        get_stt(settings.stt_provider, settings),
        get_llm(settings.llm_provider, settings),
        get_tts(settings.tts_provider, settings),
    )
