"""Construct the embedding model.

One function, because the only interesting decisions here are *when* the client
is built and *what happens when the key is missing*.

**Built lazily, never at import.** A module-level `OpenAIEmbeddings()` would
make importing this package require credentials — the exact problem that made
the previous build untestable.

**The key is validated here, not at the API call.** `Settings` keeps
`openai_api_key` optional so the package imports without credentials, so
something has to enforce presence at the point of use. Doing it here turns a
missing key into a `ConfigurationError` naming the variable, instead of an
opaque 401 raised from inside the OpenAI client several frames down.

**What LangChain abstracts:** batching, retries with backoff, and the
`.embed_documents()` / `.embed_query()` split. Worth knowing that the split is
not cosmetic — some providers use different prefixes or even different models
for queries versus documents, and calling the wrong one silently degrades
retrieval. OpenAI treats them identically today.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from secfiler_rag.config import Settings, get_settings
from secfiler_rag.core.exceptions import ConfigurationError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build the configured OpenAI embedding model.

    Args:
        settings: Override the process settings. Useful in tests.

    Returns:
        A LangChain `Embeddings` instance.

    Raises:
        ConfigurationError: If `OPENAI_API_KEY` is not set.
    """
    settings = settings or get_settings()

    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set — required for embeddings. "
            "Add it to your .env file (see .env.example)."
        )

    log.debug(
        "building embedding model",
        extra={
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "batch_size": settings.embedding_batch_size,
        },
    )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        # `text-embedding-3-*` supports Matryoshka truncation: asking for fewer
        # dimensions returns a shorter vector that is still usable, trading a
        # little accuracy for memory. Pinned explicitly so the collection's
        # configured size and the model's output can never drift apart.
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
        # LangChain's name for the embedding batch size.
        chunk_size=settings.embedding_batch_size,
    )
