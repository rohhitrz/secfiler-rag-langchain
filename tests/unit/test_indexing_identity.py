"""Point-ID contract: stable, collision-free, and derived from real metadata.

These properties are what make re-indexing idempotent. If any of them breaks,
the failure is silent — the corpus quietly doubles, or one company's chunk
overwrites another's.
"""

import pytest
from langchain_core.documents import Document

from secfiler_rag.core.exceptions import IndexingError
from secfiler_rag.indexing.indexer import _document_point_id, point_id


def test_same_chunk_always_yields_the_same_id():
    """The property that makes re-indexing an overwrite rather than an append."""
    assert point_id("aapl", 42) == point_id("aapl", 42)


def test_id_is_a_valid_uuid():
    from uuid import UUID

    assert UUID(point_id("aapl", 42)).version == 5


def test_same_chunk_id_differs_across_companies():
    """chunk_id restarts at 0 per filing, so company must be part of the key."""
    assert point_id("aapl", 0) != point_id("tsla", 0)


def test_separator_prevents_key_collisions():
    """Without the '-', ('aapl1', 2) and ('aapl', 12) would collapse together."""
    assert point_id("aapl1", 2) != point_id("aapl", 12)


def test_ids_are_unique_across_a_realistic_corpus():
    ids = {point_id(company, i) for company in ("aapl", "msft", "tsla") for i in range(600)}

    assert len(ids) == 1800


def test_document_id_uses_metadata():
    doc = Document(page_content="x", metadata={"company": "msft", "chunk_id": 7})

    assert _document_point_id(doc) == point_id("msft", 7)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"company": "aapl"},
        {"chunk_id": 3},
        {"company": "aapl", "chunk_id": "3"},  # string id would hash differently
        {"company": None, "chunk_id": 3},
    ],
)
def test_missing_or_wrong_typed_metadata_raises(metadata):
    with pytest.raises(IndexingError, match="stable point ID"):
        _document_point_id(Document(page_content="x", metadata=metadata))
