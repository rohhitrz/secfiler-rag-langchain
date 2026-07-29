"""Compose load → clean → split into one callable ingestion pipeline.

The composition lives in its own module so the three stages stay independently
testable: `clean_html` takes a string and returns a string, `split_filing`
takes a string and returns Documents. Neither knows about the filesystem, and
neither reads configuration.

This module is the only place in the package that touches both `Settings` and
the disk, which is what makes the pieces above it reusable from a test, a
notebook, or a future API without dragging config along.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from secfiler_rag.config import get_settings
from secfiler_rag.core.logging import get_logger
from secfiler_rag.ingestion.cleaner import clean_html
from secfiler_rag.ingestion.loader import FilingSource, discover_filings, read_filing
from secfiler_rag.ingestion.splitter import split_filing

log = get_logger(__name__)


def ingest_filing(
    source: FilingSource,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Run one filing through the full pipeline.

    Args:
        source: The filing to ingest.
        chunk_size: Override the configured chunk size.
        chunk_overlap: Override the configured overlap.

    Returns:
        Chunked `Document`s for this filing.

    Raises:
        IngestionError: If the file cannot be read, cleaned, or split.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    html = read_filing(source)
    text = clean_html(html)
    documents = split_filing(
        text,
        company=source.company,
        source=source.source,
        chunk_size=size,
        chunk_overlap=overlap,
    )

    log.info(
        "ingested filing",
        extra={
            "company": source.company,
            "fiscal_year": source.fiscal_year,
            "raw_bytes": len(html),
            "clean_chars": len(text),
            "chunks": len(documents),
        },
    )
    return documents


def ingest_all(
    raw_dir: Path | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Ingest every filing in the raw data directory.

    Chunk IDs restart at 0 for each company, so a chunk is identified by the
    pair `(company, chunk_id)` — never by `chunk_id` alone. That pair is the
    same identity used for Qdrant point IDs and for fusion, and collapsing it
    to a bare integer silently merges one company's chunk with another's.

    Args:
        raw_dir: Directory of raw filings. Defaults to the configured location.
        chunk_size: Override the configured chunk size.
        chunk_overlap: Override the configured overlap.

    Returns:
        Documents from every filing, ordered by company.

    Raises:
        IngestionError: If the directory is missing or any filing fails.
    """
    settings = get_settings()
    directory = raw_dir if raw_dir is not None else settings.raw_data_dir

    documents: list[Document] = []
    for source in discover_filings(directory):
        documents.extend(ingest_filing(source, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    log.info(
        "ingestion complete",
        extra={"directory": str(directory), "total_chunks": len(documents)},
    )
    return documents
