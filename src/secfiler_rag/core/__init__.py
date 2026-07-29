"""Cross-cutting primitives shared by every pipeline stage.

Contains only things that have no RAG semantics of their own: logging setup and
the exception hierarchy. `core` must never import from `ingestion`, `indexing`,
`retrieval` or `generation` — that would create an import cycle and destroy the
one-way dependency rule.
"""

from secfiler_rag.core.exceptions import (
    ConfigurationError,
    IndexingError,
    IngestionError,
    SecfilerRagError,
)
from secfiler_rag.core.logging import configure_logging, get_logger

__all__ = [
    "ConfigurationError",
    "IndexingError",
    "IngestionError",
    "SecfilerRagError",
    "configure_logging",
    "get_logger",
]
