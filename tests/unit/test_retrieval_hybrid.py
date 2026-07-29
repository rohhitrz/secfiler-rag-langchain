"""Hybrid retrieval: the candidate funnel and complementary failure modes."""

from langchain_core.documents import Document

from secfiler_rag.retrieval.hybrid import HybridRetriever


def doc(chunk_id, company="aapl"):
    return Document(
        page_content=f"{company} chunk {chunk_id}",
        metadata={"company": company, "chunk_id": chunk_id},
    )


class FakeRetriever:
    """Records what it was asked for, so the funnel can be asserted on."""

    def __init__(self, ranked_ids):
        self._ranked_ids = ranked_ids
        self.calls = []

    def search(self, query, filters=None, top_k=None):
        self.calls.append((query, dict(filters or {}), top_k))
        return [doc(i) for i in self._ranked_ids[: top_k or len(self._ranked_ids)]]


def test_each_retriever_is_asked_for_the_wider_candidate_pool():
    """The measured lesson: the answer sat at rank 6 in the dense baseline, so
    asking each retriever for only the final top_k loses it before fusion."""
    dense, sparse = FakeRetriever(range(20)), FakeRetriever(range(20))

    HybridRetriever([dense, sparse], candidate_k=10, default_top_k=3).search("q")

    assert dense.calls[0][2] == 10
    assert sparse.calls[0][2] == 10


def test_final_slice_happens_after_fusion():
    dense, sparse = FakeRetriever(range(20)), FakeRetriever(range(20))

    results = HybridRetriever([dense, sparse], candidate_k=10, default_top_k=3).search("q")

    assert len(results) == 3


def test_candidate_pool_never_narrower_than_the_requested_top_k():
    dense = FakeRetriever(range(20))

    HybridRetriever([dense], candidate_k=5).search("q", top_k=12)

    assert dense.calls[0][2] == 12


def test_a_chunk_only_one_retriever_found_can_still_surface():
    """The whole point of hybrid: complementary failure modes."""
    dense = FakeRetriever([1, 2, 3, 4, 5, 42])  # the answer, buried at rank 6
    sparse = FakeRetriever([42, 9, 8])  # exact-term match puts it first

    results = HybridRetriever([dense, sparse], candidate_k=10, default_top_k=3).search("q")

    assert results[0].metadata["chunk_id"] == 42


def test_filters_are_forwarded_unchanged_to_every_retriever():
    dense, sparse = FakeRetriever([1]), FakeRetriever([1])

    HybridRetriever([dense, sparse]).search("q", {"company": "tsla"})

    assert dense.calls[0][1] == {"company": "tsla"}
    assert sparse.calls[0][1] == {"company": "tsla"}


def test_results_carry_fusion_scores_and_source_ranks():
    results = HybridRetriever([FakeRetriever([1, 2]), FakeRetriever([2, 1])]).search("q")

    assert all("score" in d.metadata for d in results)
    assert all("rrf_ranks" in d.metadata for d in results)


def test_a_single_retriever_still_works():
    results = HybridRetriever([FakeRetriever([1, 2, 3])], default_top_k=2).search("q")

    assert [d.metadata["chunk_id"] for d in results] == [1, 2]


def test_one_retriever_returning_nothing_does_not_break_the_other():
    results = HybridRetriever([FakeRetriever([1, 2]), FakeRetriever([])]).search("q")

    assert [d.metadata["chunk_id"] for d in results] == [1, 2]


def test_search_fn_adapter_matches_the_harness_signature():
    hybrid = HybridRetriever([FakeRetriever([1, 2, 3])])

    assert len(hybrid.as_search_fn()("q", {}, 2)) == 2
