"""VA-19 — typed, fail-fast configuration loader."""
import pytest

from app.config import (
    ConfigError,
    Environment,
    LogLevel,
    Settings,
    get_settings,
    load_settings,
)

# Keys the loader reads; cleared before every test so ambient shell env can't leak in.
_ENV_KEYS = ("ENVIRONMENT", "PORT", "LOG_LEVEL", "API_PREFIX", "JWT_SECRET_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build(**overrides) -> Settings:
    """Load settings ignoring any developer .env, so tests are deterministic."""
    overrides.setdefault("_env_file", None)
    return load_settings(**overrides)


# --- happy path -------------------------------------------------------------------------

def test_defaults_are_typed():
    s = build()
    assert s.app_name == "voice-ai-agent"
    assert s.environment is Environment.LOCAL
    assert s.port == 8080
    assert s.log_level is LogLevel.INFO
    assert s.api_prefix == "/api/v1"


def test_env_overrides_are_read_and_case_insensitive(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("ENVIRONMENT", "DEV")   # upper-case accepted
    monkeypatch.setenv("LOG_LEVEL", "debug")   # lower-case accepted
    monkeypatch.setenv("JWT_SECRET_KEY", "s3cret")  # required in dev
    s = load_settings(_env_file=None)
    assert s.port == 9000
    assert s.environment is Environment.DEV
    assert s.log_level is LogLevel.DEBUG


@pytest.mark.parametrize(
    "raw,expected",
    [("api/v1", "/api/v1"), ("/api/v1/", "/api/v1"), ("v2", "/v2"), ("/", "/"), ("", "/")],
)
def test_api_prefix_is_normalised(raw, expected):
    assert build(api_prefix=raw).api_prefix == expected


# --- fail-fast --------------------------------------------------------------------------

def test_invalid_port_fails_fast():
    with pytest.raises(ConfigError) as ei:
        build(port=70000)
    assert "port" in str(ei.value)


def test_invalid_log_level_fails_fast():
    with pytest.raises(ConfigError) as ei:
        build(log_level="TRACE")
    assert "log_level" in str(ei.value)


def test_missing_required_secret_in_prod_fails_fast():
    with pytest.raises(ConfigError) as ei:
        build(environment="prod")  # no jwt_secret_key
    msg = str(ei.value)
    assert "jwt_secret_key" in msg
    assert "prod" in msg


def test_prod_with_required_secret_loads():
    s = build(environment="prod", jwt_secret_key="s3cret")
    assert s.environment is Environment.PROD


def test_local_does_not_require_secret():
    # The default (local) environment boots without any secret configured.
    assert build().environment is Environment.LOCAL


# --- secret safety ----------------------------------------------------------------------

def test_secret_never_leaks_in_repr_or_str():
    s = build(jwt_secret_key="topsecret")
    assert "topsecret" not in repr(s)
    assert "topsecret" not in str(s)


def test_public_dict_redacts_secret_but_reports_presence():
    unset = build().public_dict()
    assert unset["jwt_secret_key_configured"] is False
    assert set(unset) == {
        "app_name",
        "environment",
        "port",
        "log_level",
        "api_prefix",
        "jwt_secret_key_configured",
        "allowed_origins",  # VA-16 — origins are configuration, not a secret
    }

    with_secret = build(jwt_secret_key="topsecret").public_dict()
    assert with_secret["jwt_secret_key_configured"] is True
    assert "topsecret" not in str(with_secret)


# --- caching ----------------------------------------------------------------------------

def test_get_settings_is_cached():
    first = get_settings()
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings() is not first
