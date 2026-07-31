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

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # --- Ingestion ---------------------------------------------------------
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Target characters per chunk. Baseline carried over from the "
        "previous build; change only with an eval number to justify it.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Characters shared between neighbouring chunks, so a fact split "
        "across a boundary survives in at least one of them.",
    )

    # --- Providers ---------------------------------------------------------
    # Optional at load time so the package imports and unit-tests without
    # credentials. Required at point of use — `build_embeddings()` raises
    # ConfigurationError rather than letting a None key reach the API client.
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI key for embeddings and generation.",
    )

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model. Cheap, strong baseline; "
        "a finance-tuned model is a measured upgrade, not an assumption.",
    )
    embedding_dimensions: int = Field(
        default=1536,
        gt=0,
        description="Vector width. Must match the collection's configured size — "
        "a mismatch is rejected at startup, not at upsert time.",
    )
    embedding_batch_size: int = Field(
        default=100,
        gt=0,
        description="Chunks per embedding API call. Trades request count against "
        "blast radius when one request fails.",
    )

    # --- Generation --------------------------------------------------------
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model for answer generation.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="0 for extractive, grounded answers. Higher temperature buys "
        "variety we do not want when every claim must trace to a filing.",
    )
    max_context_tokens: int = Field(
        default=4000,
        gt=0,
        description="Token ceiling for assembled context. Budgeted explicitly rather "
        "than hoped for — silent truncation drops the chunk retrieval worked to find.",
    )

    # --- Reranking ---------------------------------------------------------
    cohere_api_key: SecretStr | None = Field(
        default=None,
        description="Cohere key for cross-encoder reranking. Optional — reranking "
        "improves ordering but the system answers without it.",
    )
    rerank_model: str = Field(
        default="rerank-v3.5",
        description="Cohere rerank model.",
    )
    rerank_candidate_k: int = Field(
        default=10,
        gt=0,
        description="Candidates handed to the reranker. Wider costs more per query; "
        "a chunk not in this pool can never be recovered.",
    )
    rerank_top_k: int = Field(
        default=3,
        gt=0,
        description="Chunks kept after reranking, i.e. what reaches the LLM.",
    )

    # --- Vector store ------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Where the Qdrant server is reachable.",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Required only when Qdrant runs with auth enabled.",
    )
    qdrant_collection: str = Field(
        default="filings",
        min_length=1,
        description="Single collection for every company; scoping is a payload "
        "filter, not a separate collection.",
    )
    qdrant_timeout: int = Field(
        default=60,
        gt=0,
        description="Seconds before a Qdrant call is abandoned. Batch upserts of "
        "hundreds of points exceed the client's short default.",
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

    @model_validator(mode="after")
    def _check_overlap_fits_chunk(self) -> Settings:
        """Reject an overlap that is not smaller than the chunk size.

        At `overlap >= chunk_size` the splitter's stride becomes zero or
        negative — it stops advancing through the text. Caught here at startup
        rather than as a hang or a memory blow-up mid-ingestion.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

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
