"""Hybrid retrieval — dense and sparse, fused by rank.

**The candidate-width funnel is the design here**, and it is easy to get
subtly wrong:

```
dense  top-N ┐
             ├── RRF ──► top-N fused ──► (later: rerank) ──► top-k answer
sparse top-N ┘
```

Each retriever contributes `candidate_k` results — deliberately wider than the
final `top_k`. The measured reason: in the dense baseline, the chunk answering
the "derivative instruments" query sat at **rank 6**. Ask each retriever for 5
and fusion never sees it; ask for 10 and it survives into the fused list where
the other retriever's opinion can lift it.

**The rule that follows: never slice to the final `top_k` before fusion (or,
later, before reranking).** A chunk cut early cannot be recovered by any
downstream stage, however good.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from secfiler_rag.core.logging import get_logger
from secfiler_rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

log = get_logger(__name__)


class Retriever(Protocol):
    """The single method hybrid retrieval needs from any strategy.

    A Protocol rather than a base class: dense and sparse share no
    implementation, only this shape. It also means a future retriever — Qdrant
    sparse vectors, a knowledge-graph lookup — plugs in without touching this
    file.
    """

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Return ranked documents for a query."""
        ...


class HybridRetriever:
    """Dense + sparse retrieval combined with Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        candidate_k: int = 10,
        rrf_k: int = DEFAULT_RRF_K,
        default_top_k: int = 5,
    ) -> None:
        """Configure the funnel.

        Args:
            retrievers: Strategies to fuse. Order is irrelevant — RRF is
                symmetric, which is part of why it is easy to reason about.
            candidate_k: Results requested from each retriever before fusion.
                Must exceed the final `top_k` or fusion has no room to work.
            rrf_k: RRF damping constant.
            default_top_k: Results returned when a caller does not specify.
        """
        self._retrievers = list(retrievers)
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve from every strategy and fuse the rankings.

        Args:
            query: The question.
            filters: Metadata constraints, applied by each retriever in its own
                way — pushed into Qdrant for dense, applied in Python for
                sparse.
            top_k: Number of fused results to return.

        Returns:
            Fused documents, each carrying its RRF `score` and `rrf_ranks`.
        """
        k = top_k if top_k is not None else self._default_top_k
        # Always retrieve at least as wide as we intend to return, so a small
        # candidate_k can never silently truncate the answer set.
        candidate_k = max(self._candidate_k, k)

        result_lists = [
            retriever.search(query, filters, candidate_k) for retriever in self._retrievers
        ]

        fused = reciprocal_rank_fusion(result_lists, k=self._rrf_k, top_k=k)

        log.debug(
            "hybrid search",
            extra={
                "query": query,
                "candidates": [len(results) for results in result_lists],
                "fused": len(fused),
            },
        )
        return fused

    def as_search_fn(self) -> Any:
        """Adapt to the eval harness's `(query, filters, top_k)` signature."""

        def search_fn(query: str, filters: Mapping[str, Any], top_k: int) -> Sequence[Document]:
            return self.search(query, filters, top_k)

        return search_fn
