"""Loader contract: the filename convention is enforced, not inferred."""

import pytest

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.ingestion.loader import (
    discover_filings,
    parse_filing_name,
    read_filing,
)


def test_filename_yields_company_and_fiscal_year(tmp_path):
    source = parse_filing_name(tmp_path / "aapl-2025.htm")

    assert source.company == "aapl"
    assert source.fiscal_year == 2025
    assert source.source == "aapl-2025.htm"


def test_html_extension_is_accepted(tmp_path):
    assert parse_filing_name(tmp_path / "msft-2024.html").company == "msft"


@pytest.mark.parametrize(
    "name",
    [
        "AAPL-2025.htm",  # uppercase would never match a lowercase payload filter
        "aapl_2025.htm",  # wrong separator
        "aapl-25.htm",  # two-digit year
        "aapl.htm",  # no year
        "aapl-2025.pdf",  # not HTML
    ],
)
def test_convention_violations_raise(tmp_path, name):
    with pytest.raises(IngestionError, match="convention"):
        parse_filing_name(tmp_path / name)


def test_discovery_is_sorted_for_stable_chunk_ids(tmp_path):
    """Chunk IDs are positional, so directory order must not vary by machine."""
    for name in ("tsla-2025.htm", "aapl-2025.htm", "msft-2025.htm"):
        (tmp_path / name).write_text("<p>x</p>")

    assert [s.company for s in discover_filings(tmp_path)] == ["aapl", "msft", "tsla"]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        discover_filings(tmp_path / "does-not-exist")


def test_empty_directory_raises(tmp_path):
    with pytest.raises(IngestionError, match="No filings found"):
        discover_filings(tmp_path)


def test_read_returns_raw_html(tmp_path):
    path = tmp_path / "aapl-2025.htm"
    path.write_text("<p>Filing body</p>")

    assert read_filing(parse_filing_name(path)) == "<p>Filing body</p>"


def test_missing_file_raises(tmp_path):
    source = parse_filing_name(tmp_path / "aapl-2025.htm")

    with pytest.raises(IngestionError, match="Could not read"):
        read_filing(source)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "aapl-2025.htm"
    path.write_text("   \n  ")

    with pytest.raises(IngestionError, match="empty"):
        read_filing(parse_filing_name(path))


def test_undecodable_bytes_do_not_fail_the_filing(tmp_path):
    """Losing one stray byte beats failing an entire 8 MB document."""
    path = tmp_path / "aapl-2025.htm"
    path.write_bytes(b"<p>caf\xe9 revenue</p>")

    assert "revenue" in read_filing(parse_filing_name(path))
