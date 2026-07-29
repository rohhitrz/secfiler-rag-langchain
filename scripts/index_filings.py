"""Build the vector index from the raw filings.

Thin by design: parse arguments, wire the pieces, report. Every decision it
makes lives in the package, so this file has nothing worth unit-testing.

    uv run python scripts/index_filings.py
    uv run python scripts/index_filings.py --recreate
    uv run python scripts/index_filings.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

from secfiler_rag.config import get_settings
from secfiler_rag.core import SecfilerRagError, configure_logging, get_logger
from secfiler_rag.indexing import build_client, build_embeddings, count_points, index_documents
from secfiler_rag.ingestion import ingest_all

log = get_logger(__name__)


def main() -> int:
    """Run ingestion followed by indexing. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection before indexing (destructive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ingest and report chunk counts without embedding or writing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    try:
        documents = ingest_all()
        by_company: dict[str, int] = {}
        for doc in documents:
            company = doc.metadata["company"]
            by_company[company] = by_company.get(company, 0) + 1
        print(f"Ingested {len(documents):,} chunks: {by_company}")

        if args.dry_run:
            print("Dry run — nothing embedded or written.")
            return 0

        client = build_client(settings)
        embeddings = build_embeddings(settings)
        written = index_documents(
            documents,
            client=client,
            embeddings=embeddings,
            settings=settings,
            recreate=args.recreate,
        )
        total = count_points(client, settings.qdrant_collection)
        print(f"Indexed {written:,} points into {settings.qdrant_collection!r} (now {total:,}).")
    except SecfilerRagError as exc:
        # Our own errors are already phrased for a human; a traceback would add
        # noise, not information.
        log.error("indexing failed", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
