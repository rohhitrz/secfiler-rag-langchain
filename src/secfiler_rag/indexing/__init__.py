"""Stage 2 — turn `Document` objects into a searchable index.

Responsibility: embed chunks, manage the Qdrant collection lifecycle, and
upsert points with deterministic IDs so re-indexing is idempotent rather than
duplicative.

This is a batch, **write-path** concern, deliberately separate from
`retrieval`. The read path never imports this package, so a read-only service
carries no embedding credentials and no write logic.

Key invariants:

* Point ID is `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")` — stable, so
  re-indexing overwrites instead of appending.
* One collection for every company; scoping is a payload filter.
* Metadata lives under the `metadata` payload key, so filters address
  `metadata.company` (see `COMPANY_PAYLOAD_FIELD`).
"""

from secfiler_rag.indexing.collection import (
    COMPANY_PAYLOAD_FIELD,
    CONTENT_PAYLOAD_KEY,
    METADATA_PAYLOAD_KEY,
    build_client,
    count_points,
    ensure_collection,
    ensure_payload_index,
)
from secfiler_rag.indexing.embeddings import build_embeddings
from secfiler_rag.indexing.indexer import (
    build_vector_store,
    index_documents,
    point_id,
)

__all__ = [
    "COMPANY_PAYLOAD_FIELD",
    "CONTENT_PAYLOAD_KEY",
    "METADATA_PAYLOAD_KEY",
    "build_client",
    "build_embeddings",
    "build_vector_store",
    "count_points",
    "ensure_collection",
    "ensure_payload_index",
    "index_documents",
    "point_id",
]
