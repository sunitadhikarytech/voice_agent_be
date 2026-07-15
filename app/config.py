"""Typed, fail-fast application configuration.

VA-19 turns the day-one settings stub into a single, typed source of configuration for the
whole service. Configuration is read from environment variables (and a local ``.env``),
coerced into typed fields, and validated eagerly so that any misconfiguration surfaces as a
clear error **at startup** rather than mid-request.

Design notes
------------
* One ``Settings`` object is the only place the app reads configuration from. It is injected
  into request handlers via FastAPI dependencies (see ``app.main``).
* Secrets use :class:`pydantic.SecretStr` so they never leak into logs, ``repr`` or
  tracebacks. ``public_dict`` returns a redacted, log-safe view.
* Some keys are optional locally but **required in ``dev`` / ``prod``** (e.g. the JWT signing
  key). Missing them fails fast with a message naming exactly what is wrong. Later tickets
  extend the required set (provider keys in VA-14) and source these values from Secret
  Manager; the loader contract does not change.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment. ``local`` relaxes the required-secret checks."""

    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid.

    Carries a human-readable, multi-line message that names each offending key, so a failed
    startup is actionable instead of a raw validation dump.
    """


# Fields that must be present when the service runs outside ``local``. Later tickets append
# to this set (e.g. provider API keys in VA-14) without touching the loader.
REQUIRED_IN_CLOUD: frozenset[str] = frozenset({"jwt_secret_key"})


class Settings(BaseSettings):
    """Typed runtime configuration, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="voice-ai-agent")
    environment: Environment = Field(default=Environment.LOCAL)
    # Cloud Run injects $PORT at runtime; default to 8080 locally.
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    api_prefix: str = Field(default="/api/v1")

    # Provider selection (VA-30). Swapping a provider is a config change, not a code change.
    # These names resolve through app.providers.factory; "mock" is always available and the
    # real adapters register under these names in VA-31 (deepgram) / VA-34 (gemini) /
    # VA-43 (cartesia).
    stt_provider: str = Field(default="deepgram")
    llm_provider: str = Field(default="gemini")
    tts_provider: str = Field(default="cartesia")
    realtime_provider: str = Field(default="openai")  # fast-path, voice-to-voice (VA-46)

    # Deepgram STT (VA-31). Key optional locally; sourced from Secret Manager in VA-14.
    deepgram_api_key: SecretStr = Field(default=SecretStr(""))
    deepgram_model: str = Field(default="nova-3")

    # ElevenLabs (VA-33 alternate STT; VA-44 alternate TTS shares the key).
    elevenlabs_api_key: SecretStr = Field(default=SecretStr(""))
    elevenlabs_stt_model: str = Field(default="scribe_v2_realtime")
    elevenlabs_tts_model: str = Field(default="eleven_flash_v2_5")
    elevenlabs_voice_id: str = Field(default="")

    # Gemini LLM (VA-34). Flash-tier model; key sourced from Secret Manager in VA-14.
    google_api_key: SecretStr = Field(default=SecretStr(""))
    gemini_model: str = Field(default="gemini-2.0-flash")
    gemini_system_prompt: str = Field(
        default="You are a helpful voice assistant. Answer concisely and conversationally."
    )
    # VA-36: cache the full document as Gemini cached content so repeat turns aren't re-billed
    # for the large context. Falls back to inlining the document when disabled.
    gemini_enable_prompt_caching: bool = Field(default=True)

    # Cartesia TTS (VA-43). Sonic-class model; key sourced from Secret Manager in VA-14.
    cartesia_api_key: SecretStr = Field(default=SecretStr(""))
    cartesia_model: str = Field(default="sonic-2")
    cartesia_voice_id: str = Field(default="")

    # OpenAI Realtime (VA-46). Voice-to-voice; key sourced from Secret Manager in VA-14.
    # NOTE: the adapter speaks the beta `realtime=v1` wire protocol, so the model must be a
    # beta-protocol model (the GA `gpt-realtime` uses a different event schema).
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_realtime_model: str = Field(default="gpt-4o-realtime-preview")
    openai_voice: str = Field(default="alloy")

    # VA-49: when the realtime path fails before delivering anything, re-run the turn on the
    # traditional pipeline instead of surfacing an error. False = fail fast.
    realtime_fallback_enabled: bool = Field(default=True)

    # Full-document grounding (VA-35). The whole source document is the context — no RAG.
    # Empty locally (grounding off); when set, the file must exist and fit the window.
    source_doc_path: str = Field(default="")
    context_window_tokens: int = Field(default=1_000_000, gt=0)

    # Rolling conversation memory budget (VA-41) — distinct from the document context.
    conversation_memory_tokens: int = Field(default=2000, gt=0)

    # Secret. Optional locally, required in dev/prod (see REQUIRED_IN_CLOUD). Consumed by the
    # auth middleware in VA-15 and sourced from Secret Manager in VA-14.
    jwt_secret_key: SecretStr = Field(default=SecretStr(""))

    # Rate limiting (VA-17). Token bucket per authenticated subject (or client IP when auth
    # is off) across everything under the API prefix. 0 (the default) disables limiting —
    # local/offline development is unthrottled; deployments opt in. Burst is the bucket
    # capacity; 0 means "same as the per-minute rate".
    rate_limit_per_minute: int = Field(default=0, ge=0)
    rate_limit_burst: int = Field(default=0, ge=0)

    # CORS lockdown (VA-16). Comma-separated origins allowed to call the API from a browser
    # (e.g. "https://app.example.com,https://staging.example.com"). Empty — the default —
    # means no cross-origin access at all; the reference dashboard is served same-origin at
    # /ui and needs no entry. Wildcards are rejected: the lockdown is to *known* origins.
    allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- normalisation so env values are forgiving about case ---
    @field_validator("environment", mode="before")
    @classmethod
    def _normalise_environment(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: Any) -> Any:
        return v.upper() if isinstance(v, str) else v

    @field_validator("api_prefix")
    @classmethod
    def _clean_api_prefix(cls, v: str) -> str:
        v = "/" + v.strip().strip("/")
        return v.rstrip("/") or "/"

    @field_validator("jwt_secret_key")
    @classmethod
    def _jwt_secret_strength(cls, v: SecretStr) -> SecretStr:
        """When a signing key is set it must be usable for HS256: RFC 7518 §3.2 requires a
        key of at least the hash size (256 bits). A short key is a brute-forceable one, so
        fail fast rather than run with weak auth."""
        secret = v.get_secret_value()
        if secret and len(secret.encode()) < 32:
            raise ValueError(
                "jwt_secret_key must be at least 32 bytes for HS256 (RFC 7518 §3.2); "
                "generate one with: openssl rand -hex 32"
            )
        return v

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: Any) -> Any:
        """Accept the natural env form — a comma-separated string — as well as a list."""
        if isinstance(v, str):
            return [part for part in (piece.strip() for piece in v.split(",")) if part]
        return v

    @field_validator("allowed_origins")
    @classmethod
    def _validate_origins(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for origin in v:
            if "*" in origin:
                raise ValueError(
                    "wildcard origins are not allowed — list each known origin explicitly"
                )
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"origin must include its scheme (http/https): {origin!r}")
            cleaned.append(origin.rstrip("/"))
        return cleaned

    @model_validator(mode="after")
    def _require_secrets_outside_local(self) -> "Settings":
        if self.environment is not Environment.LOCAL:
            missing = sorted(
                name
                for name in REQUIRED_IN_CLOUD
                if not _is_set(getattr(self, name))
            )
            if missing:
                raise ValueError(
                    f"required in environment '{self.environment.value}' but unset: "
                    + ", ".join(missing)
                )
        return self

    def public_dict(self) -> dict[str, Any]:
        """A log-safe view of the settings. Secret values are never included; secret keys
        are reported only as a configured/not-configured boolean."""
        return {
            "app_name": self.app_name,
            "environment": self.environment.value,
            "port": self.port,
            "log_level": self.log_level.value,
            "api_prefix": self.api_prefix,
            "jwt_secret_key_configured": _is_set(self.jwt_secret_key),
            # setting the key is what enables bearer-JWT auth (VA-15)
            "auth_enabled": _is_set(self.jwt_secret_key),
            "allowed_origins": list(self.allowed_origins),
            "rate_limit_per_minute": self.rate_limit_per_minute,  # 0 = off (VA-17)
        }


def _is_set(value: Any) -> bool:
    """True when a (possibly secret) value is non-empty."""
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value())
    return bool(value)


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a clear, actionable message."""
    lines = ["Invalid configuration — fix the following and restart:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(config)"
        lines.append(f"  - {loc}: {err.get('msg', 'invalid value')}")
    return "\n".join(lines)


def load_settings(**overrides: Any) -> Settings:
    """Build :class:`Settings`, converting validation failures into a clear
    :class:`ConfigError`. ``overrides`` are forwarded to the model (tests pass
    ``_env_file=None`` for isolation)."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once. Call ``get_settings.cache_clear()``
    in tests to force a reload."""
    return load_settings()
