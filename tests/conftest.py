"""Shared pytest fixtures.

The key concern here is *isolation*: the developer's real `.env` sits at the
repo root with real API keys, and pydantic-settings would happily read it during
a unit test. Tests that touch configuration must therefore start from a known,
empty environment — otherwise a passing test on your laptop fails in CI (or,
worse, passes for the wrong reason).
"""

import pytest

from secfiler_rag.config.settings import get_settings

# Every env var Settings knows about. Cleared before configuration tests so the
# assertions describe defaults, not whatever happens to be on this machine.
_SETTINGS_ENV_VARS = (
    "ENVIRONMENT",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "DATA_DIR",
    "OPENAI_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
)


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
