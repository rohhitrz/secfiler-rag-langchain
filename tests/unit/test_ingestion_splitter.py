"""Splitter contract: metadata, boundaries, overlap, and chunk identity."""

import pytest

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.ingestion.splitter import split_filing

TEXT = "\n".join(f"Line {i} of the filing with enough text to matter." for i in range(60))


def _split(text=TEXT, *, size=200, overlap=40):
    return split_filing(
        text, company="aapl", source="aapl-2025.htm", chunk_size=size, chunk_overlap=overlap
    )


def test_every_chunk_carries_the_metadata_contract():
    docs = _split()

    assert docs, "expected at least one chunk"
    for doc in docs:
        assert doc.metadata["company"] == "aapl"
        assert doc.metadata["source"] == "aapl-2025.htm"
        assert isinstance(doc.metadata["chunk_id"], int)
        assert isinstance(doc.metadata["start_index"], int)


def test_chunk_ids_are_contiguous_from_zero():
    """Positional IDs — `(company, chunk_id)` is the identity used downstream."""
    docs = _split()

    assert [d.metadata["chunk_id"] for d in docs] == list(range(len(docs)))


def test_start_indices_are_non_decreasing():
    docs = _split()

    offsets = [d.metadata["start_index"] for d in docs]
    assert offsets == sorted(offsets)


def test_start_index_points_at_the_real_location():
    docs = _split()

    for doc in docs:
        start = doc.metadata["start_index"]
        assert TEXT[start : start + len(doc.page_content)] == doc.page_content


def test_line_boundaries_are_preferred_over_blind_cuts():
    """A recursive splitter should cut on structure, not every Nth character."""
    docs = _split()

    intact = sum(1 for d in docs if d.page_content.startswith("Line "))
    assert intact > len(docs) // 2


def test_table_rows_are_not_cut_through():
    text = "\n".join(f"Region {i} | $1{i},000 | {i}% | $2{i},000" for i in range(40))

    docs = split_filing(text, company="aapl", source="f.htm", chunk_size=200, chunk_overlap=0)

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if line.strip():
                assert line.count("|") == 3, f"row was split mid-way: {line!r}"


def test_overlap_shares_text_between_neighbours():
    with_overlap = _split(overlap=80)
    without_overlap = _split(overlap=0)

    assert len(with_overlap) > len(without_overlap)


def test_zero_overlap_is_allowed():
    assert _split(overlap=0)


def test_no_chunk_greatly_exceeds_the_target_size():
    """Unsplittable runs can exceed the target; a systematic overshoot cannot."""
    docs = _split(size=200, overlap=40)

    assert max(len(d.page_content) for d in docs) <= 200


def test_empty_text_raises():
    with pytest.raises(IngestionError, match="empty text"):
        _split(text="   \n  ")
