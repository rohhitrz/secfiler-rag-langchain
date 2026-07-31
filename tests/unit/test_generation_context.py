"""Context assembly: numbering, rendering, and the token budget."""

from langchain_core.documents import Document

from secfiler_rag.generation.context import build_context, count_tokens


def doc(text="Total net sales | $416,161", company="aapl", chunk_id=12):
    return Document(
        page_content=text,
        metadata={"company": company, "chunk_id": chunk_id, "source": f"{company}-2025.htm"},
    )


def test_blocks_are_numbered_from_one():
    """Models cite small integers reliably; they cannot cite chunk IDs."""
    blocks, _ = build_context([doc(), doc(), doc()], max_tokens=10_000)

    assert [b.marker for b in blocks] == [1, 2, 3]


def test_rendered_context_carries_provenance_and_content():
    _, rendered = build_context([doc(text="net sales were 416,161")], max_tokens=10_000)

    assert "[1]" in rendered
    assert "company: aapl" in rendered
    assert "chunk: 12" in rendered
    assert "net sales were 416,161" in rendered


def test_blocks_expose_metadata_for_citation_resolution():
    blocks, _ = build_context([doc(company="tsla", chunk_id=21)], max_tokens=10_000)

    assert blocks[0].company == "tsla"
    assert blocks[0].chunk_id == 21
    assert blocks[0].source == "tsla-2025.htm"


def test_missing_metadata_degrades_without_crashing():
    blocks, rendered = build_context([Document(page_content="text", metadata={})], max_tokens=1000)

    assert blocks[0].company == "unknown"
    assert blocks[0].chunk_id is None
    assert "text" in rendered


def test_budget_drops_lower_ranked_chunks_first():
    """Input is ranked, so if something must go it should be the weakest."""
    big = doc(text="word " * 400)

    blocks, _ = build_context([big, big, big], max_tokens=600)

    assert 0 < len(blocks) < 3
    assert [b.marker for b in blocks] == list(range(1, len(blocks) + 1))


def test_everything_fits_under_a_generous_budget():
    blocks, _ = build_context([doc(), doc()], max_tokens=100_000)

    assert len(blocks) == 2


def test_rendered_context_stays_within_the_budget():
    big = doc(text="word " * 200)

    _, rendered = build_context([big] * 10, max_tokens=800)

    assert count_tokens(rendered) <= 800


def test_no_documents_yields_empty_context():
    blocks, rendered = build_context([], max_tokens=1000)

    assert blocks == []
    assert rendered == ""


def test_token_counting_is_nonzero_and_monotonic():
    assert count_tokens("hello") > 0
    assert count_tokens("hello world again") > count_tokens("hello")
