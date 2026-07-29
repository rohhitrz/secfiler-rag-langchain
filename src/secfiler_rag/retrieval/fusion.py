"""Reciprocal Rank Fusion — combining rankings from different retrievers.

**The problem fusion solves.** BM25 scores are unbounded and corpus-dependent
(roughly 0 to 13 here). Cosine similarity is bounded (roughly 0 to 1). Adding them
means BM25 silently dominates; normalising them first requires per-corpus
tuning that breaks whenever the corpus changes, and a blend weight that has to be
re-tuned whenever either retriever changes.

**RRF sidesteps all of it by throwing the scores away and keeping only rank:**

```
score(doc) = Σ over retrievers  1 / (k + rank_in_that_retriever)
```

Rank is scale-free, so this works with any retriever — including ones with no
meaningful score at all. It is also why a new retriever can be added later
without re-tuning anything.

**What `k` does.** At k=60, rank 1 contributes 1/61 ≈ 0.0164 and rank 2
contributes 1/62 ≈ 0.0161 — nearly identical. A large `k` flattens the curve,
so being top-1 in one retriever is worth less than appearing respectably in
*both*. That is the intended behaviour: agreement between independent
retrievers is stronger evidence than one retriever's confidence. A small `k`
(say 1) would let a single confident retriever dominate the fused list.

**Identity is `(company, chunk_id)`.** `chunk_id` alone restarts per filing, so
using it would merge Apple's chunk 42 with Tesla's — the same reasoning as the
Qdrant point-ID scheme.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Document]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> list[Document]:
    """Fuse several ranked lists into one.

    Args:
        result_lists: One ranked list per retriever.
        k: Damping constant. Larger flattens the contribution of top ranks,
            rewarding agreement between retrievers over one retriever's
            confidence.
        top_k: Number of fused results to return.

    Returns:
        Documents ranked by fused score. Each carries `score` (the RRF score,
        overwriting whatever the source retriever wrote) and `rrf_ranks` —
        the per-retriever ranks that produced it, for debugging.

    Raises:
        RetrievalError: If `k` is not positive, which would divide by zero or
            invert the ranking.
    """
    if k <= 0:
        raise RetrievalError(f"RRF k must be positive, got {k}")

    totals: dict[tuple[object, object], float] = defaultdict(float)
    ranks: dict[tuple[object, object], dict[int, int]] = defaultdict(dict)
    documents: dict[tuple[object, object], Document] = {}

    for retriever_index, results in enumerate(result_lists):
        for rank, document in enumerate(results, start=1):
            key = _identity(document)
            totals[key] += 1.0 / (k + rank)
            ranks[key][retriever_index] = rank
            # First writer wins: retrievers return equivalent copies of the
            # same chunk, so this only decides whose metadata object is reused.
            documents.setdefault(key, document)

    # Ties are common — two documents at the same rank in different retrievers
    # score identically. Sorting on score alone leaves the order to dict
    # insertion, which depends on the order retrievers were passed in, so
    # fusing [dense, sparse] and [sparse, dense] could return different
    # rankings. Breaking ties on the identity key makes fusion deterministic
    # and genuinely symmetric.
    ordered = sorted(totals.items(), key=lambda pair: (-pair[1], str(pair[0])))

    fused = [
        Document(
            page_content=documents[key].page_content,
            # Every stage that reorders results must overwrite `score` with its
            # own number, or a printed ranking shows scores that do not explain
            # the order.
            metadata={**documents[key].metadata, "score": score, "rrf_ranks": dict(ranks[key])},
        )
        for key, score in ordered[:top_k]
    ]

    log.debug(
        "rrf fusion",
        extra={"inputs": len(result_lists), "unique": len(totals), "returned": len(fused)},
    )
    return fused


def _identity(document: Document) -> tuple[object, object]:
    """Composite key identifying a chunk across retrievers.

    Falls back to page content when metadata is absent, so fusion still
    de-duplicates rather than double-counting a chunk that arrived without
    identity metadata.
    """
    metadata = document.metadata
    if "company" in metadata and "chunk_id" in metadata:
        return (metadata["company"], metadata["chunk_id"])
    return ("__content__", document.page_content)
