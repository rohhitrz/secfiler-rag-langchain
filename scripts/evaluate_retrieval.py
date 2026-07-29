"""Measure dense retrieval against an evaluation dataset.

    uv run python scripts/evaluate_retrieval.py                 # live Qdrant
    uv run python scripts/evaluate_retrieval.py --in-memory     # ephemeral index
    uv run python scripts/evaluate_retrieval.py --top-k 3 --audit

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
from secfiler_rag.retrieval import DenseRetriever

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

        if args.in_memory:
            client = QdrantClient(location=":memory:")
            documents = ingest_all()
            print(f"Embedding {len(documents):,} chunks (this calls the API)...")
            index_documents(documents, client=client, embeddings=embeddings, settings=settings)
        else:
            client = build_client(settings)

        store = build_vector_store(client, embeddings, collection_name=settings.qdrant_collection)
        retriever = DenseRetriever(store)

        reports = [evaluate(dataset, retriever.as_search_fn(), top_k=k) for k in sorted(args.top_k)]
    except SecfilerRagError as exc:
        log.error("evaluation failed", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(reports) > 1:
        _print_sweep(reports)
    for report in reports:
        _print_report(report, audit=args.audit)
    return 0


def _print_sweep(reports: list[EvalReport]) -> None:
    """Show quality as a function of k — where widening stops paying."""
    print()
    print("  k     hit rate    MRR")
    for report in reports:
        print(f"  {report.top_k:<5} {report.hit_rate:>7.1%}  {report.mrr:>6.3f}")


def _print_report(report: EvalReport, *, audit: bool) -> None:
    """Render the report, leading with the numbers that matter."""
    print()
    print(f"=== {report.dataset_name} @ top_k={report.top_k} ===")
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
