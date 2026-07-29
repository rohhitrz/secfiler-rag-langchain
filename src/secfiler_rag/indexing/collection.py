"""Qdrant client and collection lifecycle.

**Why we create the collection ourselves rather than letting the vector store
do it.** `QdrantVectorStore` will happily auto-create a collection on first
write, inferring the vector size by embedding a probe string. That is
convenient and wrong for a system we intend to operate: it means the schema —
vector width, distance metric, payload indexes — is a side effect of whichever
code path happened to run first, and it silently succeeds against a collection
that was created with different settings last month.

Creating it explicitly makes the schema a declared, verifiable thing, and lets
us fail loudly on the one mismatch that is otherwise invisible until retrieval
quality mysteriously collapses: **an existing collection whose vector size does
not match the current embedding model.**

**Payload key nesting — a real LangChain abstraction leak.**
`QdrantVectorStore` does not store our metadata at the payload root. It writes:

```json
{"page_content": "...", "metadata": {"company": "aapl", "chunk_id": 12}}
```

So every filter and every payload index must address `metadata.company`, not
`company`. Getting this wrong produces a filter that matches nothing, with no
error — the single most common Qdrant + LangChain bug.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from secfiler_rag.config import Settings, get_settings
from secfiler_rag.core.exceptions import IndexingError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# QdrantVectorStore nests document metadata under this payload key.
METADATA_PAYLOAD_KEY = "metadata"
CONTENT_PAYLOAD_KEY = "page_content"

# Fully-qualified payload path for the company field. Everything that filters or
# indexes on company must use this constant — never the bare field name.
COMPANY_PAYLOAD_FIELD = f"{METADATA_PAYLOAD_KEY}.company"


def build_client(settings: Settings | None = None) -> QdrantClient:
    """Build a synchronous Qdrant client.

    Synchronous is correct here: indexing is a batch job outside any event
    loop. The *read* path, once it lives inside a request handler, must use
    `AsyncQdrantClient` — a sync client called from `async def` blocks the
    whole loop, which turns one slow query into a system-wide latency spike.

    Args:
        settings: Override the process settings.

    Returns:
        A configured `QdrantClient`.
    """
    settings = settings or get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None

    log.debug("connecting to qdrant", extra={"url": settings.qdrant_url})
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
        timeout=settings.qdrant_timeout,
    )


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> bool:
    """Create the collection if absent, or verify an existing one matches.

    Args:
        client: Qdrant client.
        collection_name: Collection to create or verify.
        vector_size: Embedding width the collection must be configured for.
        recreate: Drop and rebuild the collection first. Destructive.

    Returns:
        True if the collection was created, False if it already existed.

    Raises:
        IndexingError: If an existing collection's vector size does not match
            `vector_size`.
    """
    if recreate and client.collection_exists(collection_name):
        log.warning("dropping collection", extra={"collection": collection_name})
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        _verify_vector_size(client, collection_name, vector_size)
        log.debug(
            "collection verified",
            extra={"collection": collection_name, "vector_size": vector_size},
        )
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            # Cosine because OpenAI embeddings encode meaning in direction, not
            # magnitude. Euclidean would let a long document's larger norm
            # affect ranking for reasons unrelated to relevance.
            distance=qmodels.Distance.COSINE,
        ),
    )
    log.info(
        "collection created",
        extra={"collection": collection_name, "vector_size": vector_size},
    )
    return True


def ensure_payload_index(client: QdrantClient, *, collection_name: str) -> None:
    """Index the company payload field so filtering stays cheap.

    Without an index, Qdrant can only apply a filter *after* the vector search
    — so a company-scoped query still pays for scanning every other company's
    vectors, and recall degrades because the pre-filter candidate pool was
    shared. With an index, the filter narrows the search space up front.

    Idempotent: creating an existing index is a no-op that Qdrant may report as
    an error, which is swallowed deliberately.
    """
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=COMPANY_PAYLOAD_FIELD,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        log.debug(
            "payload index ensured",
            extra={"collection": collection_name, "field": COMPANY_PAYLOAD_FIELD},
        )
    except Exception as exc:
        # Broad by intent: Qdrant reports "already exists" differently across
        # server versions and local mode, and an existing index is a success,
        # not a failure. Any other cause surfaces at query time as slow filters.
        log.debug("payload index already present", extra={"detail": str(exc)})


def count_points(client: QdrantClient, collection_name: str) -> int:
    """Return the number of points currently in the collection."""
    return client.count(collection_name=collection_name, exact=True).count


def _verify_vector_size(client: QdrantClient, collection_name: str, expected: int) -> None:
    """Raise if an existing collection was built for a different vector width.

    This is the failure that is otherwise invisible: an old collection built
    with a different embedding model accepts writes of the wrong width or
    rejects them with an opaque server error, and either way retrieval returns
    nonsense. Checking at startup converts it into one clear message.
    """
    config = client.get_collection(collection_name).config.params.vectors

    if isinstance(config, qmodels.VectorParams):
        actual = config.size
    elif isinstance(config, dict) and config:  # named vectors
        actual = next(iter(config.values())).size
    else:  # pragma: no cover — a collection with no vector config is malformed
        raise IndexingError(f"Collection {collection_name!r} has no readable vector configuration")

    if actual != expected:
        raise IndexingError(
            f"Collection {collection_name!r} has vector size {actual}, but the configured "
            f"embedding model produces {expected}. Re-index with recreate=True, or point "
            f"QDRANT_COLLECTION at a different collection."
        )
