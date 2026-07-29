"""Measure retrieval strategies against an evaluation dataset.

    uv run python scripts/evaluate_retrieval.py                       # live Qdrant
    uv run python scripts/evaluate_retrieval.py --in-memory           # ephemeral index
    uv run python scripts/evaluate_retrieval.py --in-memory --strategy dense sparse hybrid

`--strategy` scores several retrievers against **one** index build and **one**
harness, which is what makes their numbers comparable rather than merely
adjacent. Comparing runs from separate invocations would confound the
comparison with anything that changed between them.

`--in-memory` ingests and indexes into an in-process Qdrant, so a baseline can
be produced without a running server. It still calls the real embedding API,
because a number produced with fake embeddings measures nothing.

`--audit` prints the matching chunk for every passing item. Use it. A loose
expected substring can match dozens of chunks and report a hit while retrieval
actually missed — a green number nobody has read is not evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from secfiler_rag.config import get_settings
from secfiler_rag.core import SecfilerRagError, configure_logging, get_logger
from secfiler_rag.evaluation import EvalReport, evaluate, load_dataset
from secfiler_rag.indexing import (
    build_client,
    build_embeddings,
    build_vector_store,
    index_documents,
)
from secfiler_rag.ingestion import ingest_all
from secfiler_rag.retrieval import DenseRetriever, HybridRetriever, SparseRetriever

log = get_logger(__name__)

DEFAULT_DATASET = Path("evals/datasets/seed_eval_set.json")


def main() -> int:
    """Run the evaluation. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5],
        help="One or more k values. Sweeping several in one run reuses a single "
        "index build, and the curve shows where widening stops paying.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Build a throwaway index in-process instead of using a live Qdrant server.",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["dense", "sparse", "hybrid"],
        default=["dense"],
        help="Strategies to score. All share one harness and one index build, "
        "so the numbers are directly comparable.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Results each retriever contributes to hybrid fusion before slicing.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print the matching chunk for every hit, so passes can be verified.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    try:
        dataset = load_dataset(args.dataset)
        embeddings = build_embeddings(settings)
        documents = None

        if args.in_memory:
            client = QdrantClient(location=":memory:")
            documents = ingest_all()
            print(f"Embedding {len(documents):,} chunks (this calls the API)...")
            index_documents(documents, client=client, embeddings=embeddings, settings=settings)
        else:
            client = build_client(settings)

        store = build_vector_store(client, embeddings, collection_name=settings.qdrant_collection)
        strategies = _build_strategies(args, store, documents)

        # One index build, one dataset, one harness — so the numbers across
        # strategies are directly comparable rather than merely adjacent.
        reports = {
            name: [evaluate(dataset, strategy.as_search_fn(), top_k=k) for k in sorted(args.top_k)]
            for name, strategy in strategies.items()
        }
    except SecfilerRagError as exc:
        log.error("evaluation failed", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_comparison(reports)
    for name, runs in reports.items():
        for report in runs:
            _print_report(report, name=name, audit=args.audit)
    return 0


def _build_strategies(
    args: argparse.Namespace,
    store: Any,
    documents: list[Document] | None,
) -> dict[str, Any]:
    """Construct only the retrievers the requested strategies need."""
    wanted = set(args.strategy)
    built: dict[str, Any] = {}

    dense = DenseRetriever(store) if wanted & {"dense", "hybrid"} else None
    sparse = None
    if wanted & {"sparse", "hybrid"}:
        # BM25 needs the chunk texts in memory, not the vector store.
        sparse = SparseRetriever(documents if documents is not None else ingest_all())

    for name in args.strategy:
        if name == "dense":
            built["dense"] = dense
        elif name == "sparse":
            built["sparse"] = sparse
        else:
            assert dense is not None and sparse is not None  # both built above
            built["hybrid"] = HybridRetriever([dense, sparse], candidate_k=args.candidate_k)
    return built


def _print_comparison(reports: dict[str, list[EvalReport]]) -> None:
    """One table: strategies down the side, k across the top."""
    ks = [report.top_k for report in next(iter(reports.values()))]
    width = max(len(name) for name in reports) + 3

    print()
    print(f"  {'strategy':<{width}}" + "".join(f"{'k=' + str(k):<14}" for k in ks))
    print(f"  {'':<{width}}" + "hit    MRR    " * len(ks))
    for name, runs in reports.items():
        cells = "".join(f"{r.hit_rate:>5.1%}  {r.mrr:>5.3f}  " for r in runs)
        print(f"  {name:<{width}}{cells}")


def _print_report(report: EvalReport, *, name: str, audit: bool) -> None:
    """Render one strategy's report, leading with the numbers that matter."""
    print()
    print(f"=== {name} | {report.dataset_name} @ top_k={report.top_k} ===")
    hits = sum(r.hit for r in report.results)
    print(f"  hit rate : {report.hit_rate:.1%} ({hits}/{len(report.results)})")
    print(f"  MRR      : {report.mrr:.3f}")
    print(f"  latency  : {report.median_latency_ms:.0f} ms (median)")

    for tier in (1, 2):
        subset = report.by_tier(tier)
        if subset.results:
            label = "lexical smoke test" if tier == 1 else "natural language"
            print(
                f"  tier {tier}   : {subset.hit_rate:.1%} "
                f"({sum(r.hit for r in subset.results)}/{len(subset.results)})  — {label}"
            )

    if report.misses:
        print("\n--- misses ---")
        for result in report.misses:
            print(f"  ✗ {result.item.query!r}")
            print(f"      expected: {result.item.expected_substring!r}")
            print(f"      filters : {dict(result.item.filters)}")
            for rank, doc in enumerate(result.retrieved[:3], start=1):
                preview = " ".join(doc.page_content.split())[:110]
                print(f"      #{rank} chunk {doc.metadata.get('chunk_id')}: {preview}")

    if audit:
        print("\n--- audit (verify each pass is real, not a substring accident) ---")
        for result in report.results:
            if not result.hit:
                continue
            print(f"  ✓ {result.item.query!r}")
            print(f"      rank {result.matched_rank}, chunk {result.matched_chunk_id}")
            print(f"      {result.matched_excerpt()!r}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
