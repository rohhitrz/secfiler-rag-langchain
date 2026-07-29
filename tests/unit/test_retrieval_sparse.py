"""Sparse retrieval: tokenizer symmetry, exact-term strength, filtering."""

import pytest
from langchain_core.documents import Document

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.retrieval.sparse import SparseRetriever, tokenize

CORPUS = [
    ("aapl", 0, "The Company uses derivative instruments to manage foreign currency risk."),
    ("aapl", 1, "All derivative instruments are recorded in the Balance Sheets at fair value."),
    ("aapl", 2, "Total net sales | $416,161 | 6% | $391,035"),
    ("tsla", 0, "Powerwall and Megapack are our lithium-ion battery energy storage products."),
    ("tsla", 1, "Vehicle deliveries increased during the year across all models."),
    ("msft", 0, "Our reportable segments are Productivity and Business Processes."),
]


@pytest.fixture
def retriever():
    return SparseRetriever(
        [
            Document(page_content=text, metadata={"company": company, "chunk_id": chunk_id})
            for company, chunk_id, text in CORPUS
        ],
        default_top_k=3,
    )


def test_tokenizer_lowercases_and_splits_on_punctuation():
    assert tokenize("Powerwall, Megapack!") == ["powerwall", "megapack"]


def test_tokenizer_is_symmetric_for_query_and_document():
    """The single most important property: index and query must tokenise the
    same way, or `Megapack` never matches `megapack`."""
    assert tokenize("Megapack") == tokenize("megapack")


def test_tokenizer_splits_numbers_out_of_currency():
    assert tokenize("$416,161") == ["416", "161"]


def test_tokenizer_drops_empty_input():
    assert tokenize("   ---   ") == []


def test_rare_exact_term_ranks_first(retriever):
    """BM25's whole reason for existing: rare tokens are decisive."""
    results = retriever.search("Megapack")

    assert results[0].metadata["company"] == "tsla"
    assert results[0].metadata["chunk_id"] == 0


def test_exact_phrase_beats_the_semantically_similar_chunk(retriever):
    """The case dense retrieval got wrong: 'uses derivative instruments'."""
    results = retriever.search("uses derivative instruments")

    assert results[0].metadata["chunk_id"] == 0


def test_results_carry_a_score_in_descending_order(retriever):
    scores = [d.metadata["score"] for d in retriever.search("derivative instruments", top_k=5)]

    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_zero_scoring_documents_are_excluded(retriever):
    """A document sharing no query term is not a weak match, it is no match."""
    results = retriever.search("Megapack", top_k=10)

    assert len(results) < len(CORPUS)
    assert all(d.metadata["score"] > 0 for d in results)


def test_company_filter_scopes_results(retriever):
    results = retriever.search("derivative instruments", {"company": "aapl"}, top_k=5)

    assert results
    assert {d.metadata["company"] for d in results} == {"aapl"}


def test_filter_excluding_everything_returns_nothing(retriever):
    assert retriever.search("Megapack", {"company": "aapl"}) == []


def test_unknown_filter_key_raises(retriever):
    with pytest.raises(RetrievalError, match="Unknown filter key"):
        retriever.search("Megapack", {"sector": "tech"})


def test_top_k_limits_results(retriever):
    assert len(retriever.search("derivative instruments the", top_k=2)) <= 2


def test_empty_query_raises(retriever):
    with pytest.raises(RetrievalError, match="empty query"):
        retriever.search("  ")


def test_query_with_no_known_terms_returns_nothing(retriever):
    assert retriever.search("zzzzz qqqqq") == []


def test_empty_corpus_raises():
    with pytest.raises(RetrievalError, match="empty corpus"):
        SparseRetriever([])


def test_original_documents_are_not_mutated(retriever):
    """Writing scores onto the caller's documents would corrupt the index."""
    retriever.search("Megapack")

    assert all("score" not in doc.metadata for doc in retriever._documents)


def test_search_fn_adapter_matches_the_harness_signature(retriever):
    results = retriever.as_search_fn()("Megapack", {"company": "tsla"}, 2)

    assert results
    assert results[0].metadata["company"] == "tsla"
