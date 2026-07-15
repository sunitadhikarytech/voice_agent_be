"""xAI Grok Voice realtime adapter (VA-50) — alternate voice-to-voice provider.

xAI exposes an **OpenAI-compatible** realtime API, so this adapter is a configuration of
the hardened OpenAI adapter (VA-46: queue-decoupled mic pump, reconnect-with-resume,
barge-in cancel frames) pointed at xAI's endpoint with xAI credentials and model names.
Select it with ``REALTIME_PROVIDER=grok``.

Everything protocol-level — session update, ``response.audio.delta`` parsing,
``response.cancel`` on interrupt — is inherited; only identity and endpoint differ. The
endpoint and model are configuration (``GROK_REALTIME_URL`` / ``GROK_REALTIME_MODEL``), so
tracking xAI's rollout is a config change.
"""
from __future__ import annotations

from app.providers.openai_realtime import OpenAIRealtime

DEFAULT_URL = "wss://api.x.ai/v1/realtime"


class GrokRealtime(OpenAIRealtime):
    """RealtimeProvider backed by xAI's OpenAI-compatible realtime WebSocket."""

    name = "grok"

    @classmethod
    def from_settings(cls, settings) -> "GrokRealtime":
        return cls(
            api_key=settings.xai_api_key.get_secret_value(),
            model=settings.grok_realtime_model,
            voice=settings.grok_voice,
            url=settings.grok_realtime_url,
        )
