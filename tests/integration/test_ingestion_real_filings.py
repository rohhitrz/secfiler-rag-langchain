"""Ingestion against the real 10-K corpus.

Marked `integration` because parsing ~12 MB of HTML takes seconds — too slow
for the save-loop suite, but essential: the unit fixtures are miniatures, and
only the real filings prove the cleaner survives production markup.

Run with `uv run pytest -m integration`. No network or Docker needed; the
filings are tracked in the repo.
"""

from pathlib import Path

import pytest

from secfiler_rag.ingestion import clean_html, discover_filings, ingest_all, read_filing

RAW_DIR = Path("data/raw")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RAW_DIR.is_dir(), reason="raw corpus not present"),
]


def test_all_three_filings_are_discovered():
    assert [s.company for s in discover_filings(RAW_DIR)] == ["aapl", "msft", "tsla"]


def test_apple_income_statement_figures_survive_cleaning():
    """The regression that motivated the cleaner rewrite.

    These are real FY2025 figures from Apple's 10-K. The previous pipeline
    deleted inline-XBRL-wrapped values, so numbers like these vanished and the
    income statement read `Products $ $ $`.
    """
    source = next(s for s in discover_filings(RAW_DIR) if s.company == "aapl")

    text = clean_html(read_filing(source))

    assert "416,161" in text  # total net sales
    assert "Apple Inc." in text  # deleted outright by the old cleaner
    assert "10-K" in text  # so was the form type
    assert "fasb.org" not in text  # taxonomy noise still excluded


def test_table_rows_keep_labels_beside_figures():
    source = next(s for s in discover_filings(RAW_DIR) if s.company == "aapl")

    lines = clean_html(read_filing(source)).split("\n")
    row = next(line for line in lines if line.startswith("Total net sales"))

    assert "416,161" in row, f"label separated from its figures: {row!r}"


def test_full_corpus_ingests_into_chunks():
    docs = ingest_all(RAW_DIR)

    companies = {d.metadata["company"] for d in docs}
    assert companies == {"aapl", "msft", "tsla"}

    pairs = {(d.metadata["company"], d.metadata["chunk_id"]) for d in docs}
    assert len(pairs) == len(docs), "chunk identity is not unique"

    assert all(d.page_content.strip() for d in docs), "empty chunk produced"
