"""Write embedded chunks into Qdrant.

**The one decision that matters here: point IDs are deterministic.**

```python
uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")
```

The same chunk always maps to the same point ID, which makes re-indexing an
*overwrite* rather than an append. With auto-generated IDs — the default in
every vector-store quickstart — running the indexer twice silently doubles the
corpus, and the symptom is not an error but a slow degradation: duplicates
crowd the top-k, so every retrieval returns the same content three times and
the reranker has nothing diverse to work with.

Two details in that key:

* **`company` must be in it.** `chunk_id` restarts at 0 for each filing, so
  IDs would collide across companies and Tesla's chunk 0 would overwrite
  Apple's.
* **The separator is load-bearing.** Without it, `("aapl1", 2)` and
  `("aapl", 12)` produce the same string.

Qdrant requires point IDs to be UUIDs or unsigned integers, so a plain string
key is not an option — `uuid5` hashes our natural key into a valid, stable UUID.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_DNS, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from secfiler_rag.config import Settings, get_settings
from secfiler_rag.core.exceptions import IndexingError
from secfiler_rag.core.logging import get_logger
from secfiler_rag.indexing.collection import (
    CONTENT_PAYLOAD_KEY,
    METADATA_PAYLOAD_KEY,
    ensure_collection,
    ensure_payload_index,
)

log = get_logger(__name__)


def point_id(company: str, chunk_id: int) -> str:
    """Derive a chunk's stable Qdrant point ID.

    Args:
        company: Lowercase ticker.
        chunk_id: Position within that company's chunk list.

    Returns:
        A UUID5 string, identical for identical inputs across processes and
        machines.
    """
    return str(uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}"))


def build_vector_store(
    client: QdrantClient,
    embeddings: Embeddings,
    *,
    collection_name: str,
) -> QdrantVectorStore:
    """Wrap an existing collection in a LangChain vector store.

    The payload keys are passed explicitly rather than left to defaults. They
    happen to match the defaults today, but every filter in the retrieval layer
    depends on this layout, so it is stated where it can be seen and changed in
    one place.
    """
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key=CONTENT_PAYLOAD_KEY,
        metadata_payload_key=METADATA_PAYLOAD_KEY,
    )


def index_documents(
    documents: Sequence[Document],
    *,
    client: QdrantClient,
    embeddings: Embeddings,
    settings: Settings | None = None,
    recreate: bool = False,
) -> int:
    """Embed documents and upsert them into Qdrant.

    Args:
        documents: Chunks from the ingestion stage.
        client: Qdrant client.
        embeddings: Embedding model.
        settings: Override the process settings.
        recreate: Drop the collection first. Destructive.

    Returns:
        Number of points upserted.

    Raises:
        IndexingError: If a document is missing the metadata needed to build a
            stable ID, or if the collection's vector size does not match.
    """
    settings = settings or get_settings()

    if not documents:
        raise IndexingError("No documents to index — ingestion produced nothing")

    ids = [_document_point_id(doc) for doc in documents]
    if len(set(ids)) != len(ids):
        raise IndexingError(
            "Duplicate point IDs — two documents share a (company, chunk_id) pair. "
            "Chunk identity is broken upstream in ingestion."
        )

    ensure_collection(
        client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimensions,
        recreate=recreate,
    )
    ensure_payload_index(client, collection_name=settings.qdrant_collection)

    store = build_vector_store(client, embeddings, collection_name=settings.qdrant_collection)

    # Batched so one failed request costs a batch, not the whole corpus — and
    # so progress is visible in the logs on a run that takes minutes.
    batch_size = settings.embedding_batch_size
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(list(batch), ids=ids[start : start + batch_size])
        log.debug(
            "indexed batch",
            extra={"from": start, "to": start + len(batch), "total": len(documents)},
        )

    log.info(
        "indexing complete",
        extra={
            "collection": settings.qdrant_collection,
            "points": len(documents),
            "model": settings.embedding_model,
        },
    )
    return len(documents)


def _document_point_id(document: Document) -> str:
    """Build one document's point ID, failing loudly on missing metadata."""
    company = document.metadata.get("company")
    chunk_id = document.metadata.get("chunk_id")

    if not isinstance(company, str) or not isinstance(chunk_id, int):
        raise IndexingError(
            "Document is missing the (company, chunk_id) metadata required for a "
            f"stable point ID: {document.metadata!r}"
        )
    return point_id(company, chunk_id)
