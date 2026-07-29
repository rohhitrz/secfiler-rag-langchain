"""The retrieval evaluation harness.

**This module is frozen by contract: it must never learn domain knowledge.**

It accepts a `SearchFn` and a dataset. It does not know whether it is scoring
BM25, dense search, or a five-stage hybrid pipeline; it does not know what a
"company" is. Filters travel from the dataset to the retriever as an opaque
mapping that the harness forwards without inspecting.

That ignorance is the entire point. The moment the harness special-cases a
strategy — "if this is BM25, lowercase the query first" — its numbers stop
being comparable across strategies, and every A/B it produces afterwards is
measuring the harness rather than the retriever.

**Why substring matching rather than chunk IDs.** Chunk IDs renumber whenever
the chunker changes, which would silently invalidate the dataset on any
ingestion tweak. A substring of the cleaned text survives re-chunking, so the
same dataset can compare a 1000-character chunker against an 800-character one.
The cost is false positives from loose substrings — which is why every result
carries the chunk that matched, so a pass can be audited rather than trusted.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from secfiler_rag.core.logging import get_logger
from secfiler_rag.evaluation.dataset import EvalDataset, EvalItem
from secfiler_rag.evaluation.metrics import hit_rate, mean_reciprocal_rank

log = get_logger(__name__)

# A retrieval strategy, reduced to its only interesting behaviour: a query and
# opaque filters go in, ranked documents come out. Anything that satisfies this
# can be evaluated — which is what makes cross-strategy comparison honest.
SearchFn = Callable[[str, Mapping[str, Any], int], Sequence[Document]]

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ItemResult:
    """The outcome for a single evaluation item.

    `matched_rank`, `matched_chunk_id` and `matched_excerpt` exist for auditing.
    A green number that has not been read is not evidence: a loose expected
    substring can match dozens of chunks and report a hit while retrieval
    actually missed.
    """

    item: EvalItem
    retrieved: tuple[Document, ...]
    matched_rank: int | None
    latency_ms: float

    @property
    def hit(self) -> bool:
        """Whether the expected text appeared in any retrieved chunk."""
        return self.matched_rank is not None

    @property
    def matched_chunk_id(self) -> int | None:
        """Chunk ID of the matching document, for auditing."""
        if self.matched_rank is None:
            return None
        metadata = self.retrieved[self.matched_rank - 1].metadata
        chunk_id = metadata.get("chunk_id")
        return chunk_id if isinstance(chunk_id, int) else None

    def matched_excerpt(self, width: int = 160) -> str | None:
        """Text around the match, so a pass can be read and judged."""
        if self.matched_rank is None:
            return None
        content = self.retrieved[self.matched_rank - 1].page_content
        position = _normalise(content).find(_normalise(self.item.expected_substring))
        start = max(0, position - width // 2)
        return content[start : start + width]


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate results across a dataset."""

    dataset_name: str
    top_k: int
    results: tuple[ItemResult, ...]

    @property
    def hit_rate(self) -> float:
        """Fraction of items whose expected text was retrieved."""
        return hit_rate([r.hit for r in self.results])

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank — rewards ranking the right chunk higher."""
        return mean_reciprocal_rank([r.matched_rank for r in self.results])

    @property
    def median_latency_ms(self) -> float:
        """Median retrieval latency, excluding harness overhead."""
        if not self.results:
            return 0.0
        latencies = sorted(r.latency_ms for r in self.results)
        return latencies[len(latencies) // 2]

    def by_tier(self, tier: int) -> EvalReport:
        """A report restricted to one tier.

        Tier 1 is a smoke test; tier 2 is the number that should actually move.
        Reporting them together hides a regression behind near-tautological
        passes.
        """
        return EvalReport(
            dataset_name=f"{self.dataset_name}[tier{tier}]",
            top_k=self.top_k,
            results=tuple(r for r in self.results if r.item.tier == tier),
        )

    @property
    def misses(self) -> tuple[ItemResult, ...]:
        """Failed items — where the diagnostic work starts."""
        return tuple(r for r in self.results if not r.hit)


def evaluate(dataset: EvalDataset, search_fn: SearchFn, *, top_k: int = 5) -> EvalReport:
    """Score a retrieval strategy against a dataset.

    Args:
        dataset: Items to evaluate.
        search_fn: Any callable taking `(query, filters, top_k)` and returning
            ranked documents. Its internals are irrelevant here by design.
        top_k: Documents to request per query.

    Returns:
        The full report, including per-item results for auditing.
    """
    results = []
    for item in dataset.items:
        started = time.perf_counter()
        retrieved = tuple(search_fn(item.query, item.filters, top_k))
        latency_ms = (time.perf_counter() - started) * 1000

        results.append(
            ItemResult(
                item=item,
                retrieved=retrieved,
                matched_rank=_first_match_rank(retrieved, item.expected_substring),
                latency_ms=latency_ms,
            )
        )

    report = EvalReport(dataset_name=dataset.name, top_k=top_k, results=tuple(results))
    log.info(
        "evaluation complete",
        extra={
            "dataset": dataset.name,
            "items": len(results),
            "top_k": top_k,
            "hit_rate": round(report.hit_rate, 4),
            "mrr": round(report.mrr, 4),
        },
    )
    return report


def _first_match_rank(documents: Sequence[Document], expected: str) -> int | None:
    """1-based rank of the first document containing the expected text.

    Comparison is case-insensitive with whitespace collapsed, because an
    expected substring is written by a human reading cleaned text — it should
    not fail on a line break that the chunker happened to introduce.
    """
    needle = _normalise(expected)
    for rank, document in enumerate(documents, start=1):
        if needle in _normalise(document.page_content):
            return rank
    return None


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for forgiving comparison."""
    return _WHITESPACE.sub(" ", text).strip().lower()
