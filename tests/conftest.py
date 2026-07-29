"""Shared pytest fixtures.

The key concern here is *isolation*: the developer's real `.env` sits at the
repo root with real API keys, and pydantic-settings would happily read it during
a unit test. Tests that touch configuration must therefore start from a known,
empty environment — otherwise a passing test on your laptop fails in CI (or,
worse, passes for the wrong reason).
"""

from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from secfiler_rag.config.settings import Settings, get_settings

# Every env var Settings knows about. Cleared before configuration tests so the
# assertions describe defaults, not whatever happens to be on this machine.
_SETTINGS_ENV_VARS = (
    "ENVIRONMENT",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "DATA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "OPENAI_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_TIMEOUT",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
)


class IsolatedSettings(Settings):
    """Settings with dotenv loading disabled.

    Overrides only `env_file`, so every field, validator and the frozen/extra
    behaviour under test are inherited unchanged. Using the real `Settings`
    class with `_env_file=None` would work too, but that private keyword is
    invisible to the type checker — a subclass keeps the suite strictly typed.
    """

    model_config = SettingsConfigDict(env_file=None)


def make_settings(**overrides: Any) -> Settings:
    """Build Settings from process env plus explicit overrides, ignoring `.env`."""
    return IsolatedSettings(**overrides)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Drop the cached settings singleton around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all project env vars so tests observe declared defaults."""
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch
