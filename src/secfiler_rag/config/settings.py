"""Typed, environment-driven application settings.

Design notes (the "why", for the record):

* **One entry point.** `get_settings()` is the only supported way to read
  configuration. No module reads `os.environ` directly, so every knob is
  discoverable in one file and typo'd env names fail loudly at startup instead
  of silently defaulting deep inside a retriever.
* **Lazy + cached.** Settings are built on first call, not at import time. The
  previous build instantiated API clients at module import, which made tests
  require live credentials just to `import`. `functools.lru_cache` gives us a
  process-wide singleton without that side effect.
* **Unprefixed env names.** `OPENAI_API_KEY` and `LANGSMITH_*` are read
  directly by the OpenAI and LangSmith SDKs from the process environment. Using
  a custom prefix would mean maintaining two names for the same secret.
* **`extra="ignore"`.** The `.env` file is shared with tools that read it
  themselves; unknown keys must not crash the app.
* **Secrets are `SecretStr`.** They never render in logs, tracebacks or
  `repr()` by accident — you have to ask for the value explicitly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]
LogFormat = Literal["console", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Every runtime knob, validated once at startup.

    Fields are added as the module that needs them lands, so this class stays a
    truthful description of what the system actually uses.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    # --- Runtime -----------------------------------------------------------
    environment: Environment = Field(
        default="local",
        description="Deployment context. Drives log format defaults and safety checks.",
    )

    # --- Logging -----------------------------------------------------------
    log_level: LogLevel = Field(
        default="INFO",
        description="Root log level for the `secfiler_rag` logger tree.",
    )
    log_format: LogFormat = Field(
        default="console",
        description="`console` for human-readable local dev, `json` for log aggregators.",
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("data"),
        description="Root of the local data directory, resolved against the process CWD.",
    )

    # --- Providers ---------------------------------------------------------
    # Optional today: no module consumes them yet. They become required
    # (validated at point of use) when the indexing module lands.
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI key for embeddings and generation.",
    )

    # --- Observability -----------------------------------------------------
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing. Off by default so the pipeline "
        "runs with no LangSmith account.",
    )
    langsmith_project: str = Field(
        default="secfiler-rag",
        description="LangSmith project that runs are grouped under.",
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith key. Required only when `langsmith_tracing` is on.",
    )

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        """Expand `~` so `DATA_DIR=~/corpora` behaves the way a user expects."""
        return value.expanduser()

    @property
    def raw_data_dir(self) -> Path:
        """Directory holding the untouched source filings."""
        return self.data_dir / "raw"

    @property
    def is_local(self) -> bool:
        """True when running on a developer machine (relaxes strict checks)."""
        return self.environment == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached, so the `.env` file is parsed once and every caller observes the same
    object. Call `get_settings.cache_clear()` in tests that need a fresh read.
    """
    return Settings()
