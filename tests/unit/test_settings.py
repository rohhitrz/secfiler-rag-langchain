"""Configuration contract: defaults, overrides, validation, secret safety."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from secfiler_rag.config.settings import Settings, get_settings


class IsolatedSettings(Settings):
    """Settings with dotenv loading disabled.

    The developer's real `.env` sits at the repo root. Reading it here would
    make these assertions depend on one machine's secrets, so the subclass
    overrides only `env_file` and inherits every field, validator and the
    frozen/extra behaviour under test.
    """

    model_config = SettingsConfigDict(env_file=None)


def _settings(**overrides: Any) -> Settings:
    """Build Settings from process env + explicit overrides only."""
    return IsolatedSettings(**overrides)


def test_defaults_are_safe_for_local_development(clean_env):
    settings = _settings()

    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.log_format == "console"
    assert settings.data_dir == Path("data")
    # Tracing must default off: the pipeline has to run with no LangSmith account.
    assert settings.langsmith_tracing is False


def test_env_vars_override_defaults(clean_env):
    clean_env.setenv("ENVIRONMENT", "production")
    clean_env.setenv("LOG_FORMAT", "json")
    clean_env.setenv("LANGSMITH_TRACING", "true")

    settings = _settings()

    assert settings.environment == "production"
    assert settings.log_format == "json"
    assert settings.langsmith_tracing is True
    assert settings.is_local is False


def test_invalid_enum_value_fails_fast(clean_env):
    clean_env.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        _settings()


def test_raw_data_dir_derives_from_data_dir(clean_env):
    settings = _settings(data_dir=Path("/corpus"))

    assert settings.raw_data_dir == Path("/corpus/raw")


def test_data_dir_expands_user_home(clean_env):
    settings = _settings(data_dir=Path("~/corpus"))

    assert "~" not in str(settings.raw_data_dir)


def test_secrets_do_not_leak_into_repr(clean_env):
    settings = _settings(openai_api_key="sk-super-secret")

    assert "sk-super-secret" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-super-secret"


def test_settings_are_immutable(clean_env):
    settings = _settings()

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"


def test_unknown_env_keys_are_ignored(clean_env):
    """The shared .env carries keys owned by other tools; they must not crash us."""
    clean_env.setenv("SOME_TOOL_SPECIFIC_KEY", "value")

    assert _settings().environment == "local"


def test_get_settings_returns_a_cached_singleton():
    assert get_settings() is get_settings()
