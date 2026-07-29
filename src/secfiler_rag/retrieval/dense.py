"""Dense (vector) retrieval over the Qdrant collection.

**What actually happens on a query**, since this is where LangChain hides the
most: the query string is embedded with the *same* model used at index time,
then Qdrant finds the nearest stored vectors by cosine similarity using its
HNSW graph — an approximate search, not an exhaustive scan. "Approximate" is
the trade: sub-linear latency in exchange for a small chance of missing a true
nearest neighbour. At this corpus size the risk is negligible; at scale it is
the knob behind every recall-versus-speed discussion.

**Symmetry is mandatory.** Query and documents must be embedded by the same
model with the same dimensions. Mixing models produces vectors in unrelated
spaces, and the failure is not an error — it is retrieval that returns
confident nonsense.

**Filters are translated here, not in the harness.** This module is the only
place that knows a filter key called `company` maps to the Qdrant payload path
`metadata.company`. Everything upstream passes filters as an opaque mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.core.logging import get_logger
from secfiler_rag.indexing.collection import METADATA_PAYLOAD_KEY

log = get_logger(__name__)

# Filter keys this retriever understands, mapped to their payload paths.
# QdrantVectorStore nests metadata, so the path is prefixed — filtering on the
# bare field name matches nothing and raises nothing.
_FILTER_FIELDS = {"company": f"{METADATA_PAYLOAD_KEY}.company"}


class DenseRetriever:
    """Vector search over a Qdrant collection, with optional payload filtering.

    Deliberately not a `BaseRetriever` subclass. LangChain's retriever
    interface takes only a query string, so per-query filters would have to be
    baked in at construction — meaning one retriever object per company, which
    the eval harness cannot express. `as_langchain_retriever()` provides the
    framework-native adapter for LCEL composition where filters are fixed.
    """

    def __init__(self, store: QdrantVectorStore, *, default_top_k: int = 5) -> None:
        self._store = store
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve the most similar chunks.

        Args:
            query: Natural-language question.
            filters: Payload constraints, e.g. `{"company": "aapl"}`. An
                unknown key raises rather than being silently ignored.
            top_k: Number of chunks to return.

        Returns:
            Documents ranked by similarity, each carrying a `score` in its
            metadata.

        Raises:
            RetrievalError: If the query is empty or a filter key is unknown.
        """
        if not query.strip():
            raise RetrievalError("Cannot retrieve for an empty query")

        k = top_k if top_k is not None else self._default_top_k
        qdrant_filter = build_filter(filters)

        scored = self._store.similarity_search_with_score(query, k=k, filter=qdrant_filter)

        documents = []
        for document, score in scored:
            # Every stage that reorders results must write its own score here,
            # so a printed ranking always shows the number that produced it.
            document.metadata["score"] = float(score)
            documents.append(document)

        log.debug(
            "dense search",
            extra={"query": query, "filters": dict(filters or {}), "results": len(documents)},
        )
        return documents

    def as_search_fn(self) -> Any:
        """Adapt to the eval harness's `(query, filters, top_k)` signature."""

        def search_fn(query: str, filters: Mapping[str, Any], top_k: int) -> Sequence[Document]:
            return self.search(query, filters, top_k)

        return search_fn

    def as_langchain_retriever(
        self, *, filters: Mapping[str, Any] | None = None, top_k: int | None = None
    ) -> Any:
        """Return a LangChain `BaseRetriever` with fixed filters.

        For LCEL chains, where the filter is known when the chain is built.
        """
        search_kwargs: dict[str, Any] = {"k": top_k if top_k is not None else self._default_top_k}
        qdrant_filter = build_filter(filters)
        if qdrant_filter is not None:
            search_kwargs["filter"] = qdrant_filter
        return self._store.as_retriever(search_kwargs=search_kwargs)


def build_filter(filters: Mapping[str, Any] | None) -> qmodels.Filter | None:
    """Translate an opaque filter mapping into a Qdrant filter.

    Args:
        filters: Filter keys and values, or None.

    Returns:
        A Qdrant `Filter`, or None when there is nothing to constrain.

    Raises:
        RetrievalError: If a key is not a supported filter field. Unknown keys
            must not be ignored — a silently dropped filter returns another
            company's chunks and looks like a retrieval quality problem.
    """
    if not filters:
        return None

    conditions: list[qmodels.Condition] = []
    for key, value in filters.items():
        field = _FILTER_FIELDS.get(key)
        if field is None:
            raise RetrievalError(f"Unknown filter key {key!r}. Supported: {sorted(_FILTER_FIELDS)}")
        conditions.append(qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value)))

    return qmodels.Filter(must=conditions)
