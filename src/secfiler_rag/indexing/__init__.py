"""Stage 2 — turn `Document` objects into a searchable index.

Responsibility: embed chunks (OpenAI embeddings), manage the Qdrant collection
lifecycle (create / recreate / verify dimensions), and upsert points with
deterministic IDs so re-indexing is idempotent rather than duplicative.

This is a batch, offline concern. It is deliberately separate from `retrieval`
so that the read path never carries write-path dependencies.

Status: not implemented yet.
"""
