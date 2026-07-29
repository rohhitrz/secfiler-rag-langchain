"""Pipeline contract: composition, config defaults, and per-company chunk IDs."""

import pytest

from secfiler_rag.config.settings import get_settings
from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.ingestion.loader import parse_filing_name
from secfiler_rag.ingestion.pipeline import ingest_all, ingest_filing

FILING = """
<head><title>drop me</title></head>
<body>
  <script>var noise = 1;</script>
  <p>Total net sales for fiscal <ix:nonFraction>2025</ix:nonFraction> were as follows.</p>
  <table>
    <tr><td>Americas</td><td>$</td><td>167,045</td><td>3</td><td>%</td></tr>
    <tr><td>Europe</td><td>111,032</td><td>10</td><td>%</td></tr>
  </table>
  <p>Additional prose about the business and its operations follows here.</p>
</body>
"""


def _write(tmp_path, name="aapl-2025.htm", html=FILING):
    path = tmp_path / name
    path.write_text(html)
    return path


def test_end_to_end_produces_documents(tmp_path):
    docs = ingest_filing(parse_filing_name(_write(tmp_path)))

    assert docs
    body = "\n".join(d.page_content for d in docs)
    assert "Americas | $167,045 | 3%" in body
    assert "2025" in body  # the XBRL-wrapped value survived
    assert "noise" not in body
    assert "drop me" not in body


def test_metadata_reaches_documents_from_the_filename(tmp_path):
    docs = ingest_filing(parse_filing_name(_write(tmp_path, "tsla-2024.htm")))

    assert {d.metadata["company"] for d in docs} == {"tsla"}
    assert {d.metadata["source"] for d in docs} == {"tsla-2024.htm"}


def test_chunk_size_defaults_come_from_settings(clean_env):
    settings = get_settings()

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_explicit_chunk_size_overrides_settings(tmp_path):
    source = parse_filing_name(_write(tmp_path))

    small = ingest_filing(source, chunk_size=120, chunk_overlap=0)
    large = ingest_filing(source, chunk_size=4000, chunk_overlap=0)

    assert len(small) > len(large)


def test_ingest_all_restarts_chunk_ids_per_company(tmp_path):
    """`chunk_id` alone is ambiguous — identity is `(company, chunk_id)`."""
    _write(tmp_path, "aapl-2025.htm")
    _write(tmp_path, "tsla-2025.htm")

    docs = ingest_all(tmp_path, chunk_size=120, chunk_overlap=0)

    by_company: dict[str, list[int]] = {}
    for doc in docs:
        by_company.setdefault(doc.metadata["company"], []).append(doc.metadata["chunk_id"])

    assert set(by_company) == {"aapl", "tsla"}
    for ids in by_company.values():
        assert ids == list(range(len(ids)))
    # The pair is unique even though the bare ids collide.
    pairs = {(d.metadata["company"], d.metadata["chunk_id"]) for d in docs}
    assert len(pairs) == len(docs)


def test_ingest_all_is_ordered_by_company(tmp_path):
    for name in ("tsla-2025.htm", "aapl-2025.htm"):
        _write(tmp_path, name)

    companies = [d.metadata["company"] for d in ingest_all(tmp_path)]

    assert companies == sorted(companies)


def test_a_single_bad_filing_fails_the_run(tmp_path):
    """Silent partial ingestion is worse than a loud failure: a missing filing
    becomes a retrieval gap that looks like a bad answer weeks later."""
    _write(tmp_path, "aapl-2025.htm")
    (tmp_path / "msft-2025.htm").write_text("<html><head><title>t</title></head></html>")

    with pytest.raises(IngestionError):
        ingest_all(tmp_path)
