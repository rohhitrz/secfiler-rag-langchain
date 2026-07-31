"""Ask a question about the filings — the full pipeline, end to end.

    uv run python scripts/ask.py "What was Apple's total net sales in 2025?"
    uv run python scripts/ask.py "What does Tesla sell?" --company tsla
    uv run python scripts/ask.py "..." --in-memory --show-context

`--in-memory` builds a throwaway index so the whole pipeline runs without a
Qdrant server. It re-embeds the corpus on every run, so use a live server for
anything but a one-off.

`--show-context` prints the exact numbered blocks the model was given. When an
answer looks wrong, this is the first thing to read — it separates a retrieval
problem from a generation problem in one step.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from qdrant_client import QdrantClient

from secfiler_rag.config import get_settings
from secfiler_rag.core import SecfilerRagError, configure_logging, get_logger
from secfiler_rag.generation import AnswerGenerator, AnswerResult, build_llm
from secfiler_rag.indexing import (
    build_client,
    build_embeddings,
    build_vector_store,
    index_documents,
)
from secfiler_rag.ingestion import ingest_all
from secfiler_rag.retrieval import DenseRetriever, RerankingRetriever, build_rerank_client

log = get_logger(__name__)


def main() -> int:
    """Answer one question. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--company", choices=["aapl", "msft", "tsla"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    try:
        if args.in_memory:
            client = QdrantClient(location=":memory:")
            documents = ingest_all()
            print(f"Embedding {len(documents):,} chunks...", file=sys.stderr)
            index_documents(
                documents,
                client=client,
                embeddings=build_embeddings(settings),
                settings=settings,
            )
        else:
            client = build_client(settings)

        store = build_vector_store(
            client, build_embeddings(settings), collection_name=settings.qdrant_collection
        )
        retriever: Any = DenseRetriever(store)
        if not args.no_rerank:
            retriever = RerankingRetriever(
                retriever,
                build_rerank_client(settings),
                model=settings.rerank_model,
                candidate_k=settings.rerank_candidate_k,
                default_top_k=settings.rerank_top_k,
            )

        generator = AnswerGenerator(
            retriever,
            build_llm(settings),
            top_k=args.top_k or settings.rerank_top_k,
            max_context_tokens=settings.max_context_tokens,
        )
        result = generator.answer(
            args.question, {"company": args.company} if args.company else None
        )
    except SecfilerRagError as exc:
        log.error("could not answer", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print(result, show_context=args.show_context)
    return 0


def _print(result: AnswerResult, *, show_context: bool) -> None:
    """Render the answer with its provenance."""
    print(f"\nQ: {result.question}\n")
    print(result.answer)

    if result.citations:
        print("\nSources:")
        for citation in result.citations:
            print(f"  [{citation.marker}] {citation.company} · {citation.source} "
                  f"· chunk {citation.chunk_id}")
    elif not result.refused:
        # Worth surfacing: an uncited answer is unverifiable, even if correct.
        print("\n⚠ No citations — this answer cannot be traced to a filing.")

    if show_context:
        print("\n--- context the model actually received ---")
        for block in result.context_blocks:
            preview = " ".join(block.document.page_content.split())[:400]
            print(f"\n[{block.marker}] {block.company} chunk {block.chunk_id}\n{preview}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
