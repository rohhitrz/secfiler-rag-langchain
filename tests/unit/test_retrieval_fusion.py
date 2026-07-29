"""RRF: scale-independence, identity, agreement-over-confidence, and k."""

import pytest
from langchain_core.documents import Document

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.retrieval.fusion import reciprocal_rank_fusion


def doc(chunk_id, company="aapl", score=None, text=None):
    metadata = {"company": company, "chunk_id": chunk_id}
    if score is not None:
        metadata["score"] = score
    return Document(page_content=text or f"{company} chunk {chunk_id}", metadata=metadata)


def ids(documents):
    return [(d.metadata["company"], d.metadata["chunk_id"]) for d in documents]


def test_document_in_both_lists_outranks_one_that_leads_a_single_list():
    """Agreement between independent retrievers beats one retriever's confidence."""
    dense = [doc(1), doc(2), doc(3)]
    sparse = [doc(9), doc(2), doc(8)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].metadata["chunk_id"] == 2


def test_scores_of_wildly_different_scales_are_irrelevant():
    """The reason RRF exists: BM25 (~0-13) and cosine (~0-1) never meet."""
    dense = [doc(1, score=0.91), doc(2, score=0.89)]
    sparse = [doc(2, score=13.4), doc(1, score=11.9)]

    flipped = reciprocal_rank_fusion(
        [[doc(1, score=0.0001), doc(2, score=0.00009)], [doc(2, score=999.0), doc(1, score=980.0)]]
    )

    assert ids(reciprocal_rank_fusion([dense, sparse])) == ids(flipped)


def test_identity_is_the_company_chunk_pair():
    """chunk_id alone would merge Apple's chunk 42 with Tesla's."""
    fused = reciprocal_rank_fusion([[doc(1, "aapl")], [doc(1, "tsla")]])

    assert len(fused) == 2


def test_same_chunk_from_both_retrievers_appears_once():
    fused = reciprocal_rank_fusion([[doc(1)], [doc(1)]])

    assert len(fused) == 1


def test_rrf_score_arithmetic():
    fused = reciprocal_rank_fusion([[doc(1)], [doc(1)]], k=60)

    assert fused[0].metadata["score"] == pytest.approx(2 / 61)


def test_score_field_is_overwritten_by_the_fusion_score():
    """Stale retriever scores would make a printed ranking unexplainable."""
    fused = reciprocal_rank_fusion([[doc(1, score=0.99)]], k=60)

    assert fused[0].metadata["score"] == pytest.approx(1 / 61)


def test_source_ranks_are_recorded_for_debugging():
    fused = reciprocal_rank_fusion([[doc(5), doc(1)], [doc(1)]])

    by_id = {d.metadata["chunk_id"]: d.metadata["rrf_ranks"] for d in fused}
    assert by_id[1] == {0: 2, 1: 1}
    assert by_id[5] == {0: 1}


def test_large_k_favours_agreement_small_k_favours_confidence():
    """k is the knob between 'both retrievers liked it' and 'one loved it'.

    Doc 1 leads one list; doc 4 is fourth in both. At k=60 the ranks are so
    flattened that appearing twice wins (2/64 > 1/61). At k=1 the top rank is
    worth far more, so the single confident result wins (1/2 > 2/5).
    """
    dense = [doc(1), doc(2), doc(3), doc(4)]
    sparse = [doc(9), doc(8), doc(7), doc(4)]

    assert reciprocal_rank_fusion([dense, sparse], k=60)[0].metadata["chunk_id"] == 4
    assert reciprocal_rank_fusion([dense, sparse], k=1)[0].metadata["chunk_id"] == 1


def test_top_k_limits_the_fused_list():
    fused = reciprocal_rank_fusion([[doc(i) for i in range(10)]], top_k=3)

    assert len(fused) == 3


def test_single_list_preserves_its_order():
    fused = reciprocal_rank_fusion([[doc(7), doc(3), doc(9)]])

    assert [d.metadata["chunk_id"] for d in fused] == [7, 3, 9]


def test_fusion_is_symmetric_in_retriever_order():
    dense = [doc(1), doc(2), doc(3)]
    sparse = [doc(3), doc(4)]

    assert ids(reciprocal_rank_fusion([dense, sparse])) == ids(
        reciprocal_rank_fusion([sparse, dense])
    )


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_retriever_does_not_break_fusion():
    fused = reciprocal_rank_fusion([[doc(1), doc(2)], []])

    assert [d.metadata["chunk_id"] for d in fused] == [1, 2]


def test_documents_without_identity_metadata_still_deduplicate():
    plain = Document(page_content="same text", metadata={})

    assert len(reciprocal_rank_fusion([[plain], [plain]])) == 1


def test_non_positive_k_raises():
    with pytest.raises(RetrievalError, match="k must be positive"):
        reciprocal_rank_fusion([[doc(1)]], k=0)
