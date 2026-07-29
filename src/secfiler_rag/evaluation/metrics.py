"""Retrieval metrics.

Two metrics, deliberately. They answer different questions, and reporting only
one hides a real effect:

* **Hit rate @ k** — did the right chunk make it into the context at all? This
  is the ceiling on answer quality: a chunk that never reaches the prompt
  cannot be used, no matter how good the model is.
* **MRR** — *where* in the list did it land? Position matters because models
  attend most reliably to the start and end of their context, so a chunk at
  rank 5 is worth less than the same chunk at rank 1.

A change can leave hit rate flat while moving MRR substantially — that is
exactly what a reranker does, and with hit rate alone it would look like the
reranker did nothing.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_rate(hits: Sequence[bool]) -> float:
    """Fraction of queries where the expected chunk was retrieved.

    Args:
        hits: One boolean per query.

    Returns:
        A value in [0, 1]; 0.0 for an empty sequence.
    """
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean of 1/rank across queries, counting a miss as 0.

    Args:
        ranks: 1-based rank of the first correct result per query, or None if
            it was not retrieved.

    Returns:
        A value in [0, 1]; 1.0 means every query put the right chunk first.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks)
