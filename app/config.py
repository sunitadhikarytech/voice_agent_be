"""Typed application settings.

VA-01 provides a minimal, typed settings object so the rest of the app has a single
place to read configuration from. It is intentionally small; VA-19 expands this into a
fail-fast loader that validates every required key at startup and reads secrets from a
secret manager. Nothing here reads real secrets — values come from the environment or a
local `.env` (see `.env.example`).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="voice-ai-agent")
    environment: str = Field(default="local", description="local | dev | prod")
    # Cloud Run injects $PORT at runtime; default to 8080 locally.
    port: int = Field(default=8080)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
