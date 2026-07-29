"""Dense retrieval: filter translation, scoping, and scores on results.

Runs against in-memory Qdrant with deterministic fake embeddings — real search,
real payload filters, no Docker.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.indexing.indexer import build_vector_store, index_documents
from secfiler_rag.retrieval.dense import DenseRetriever
from secfiler_rag.retrieval.filters import build_qdrant_filter
from tests.conftest import make_settings

VECTOR_SIZE = 32


@pytest.fixture
def retriever():
    client = QdrantClient(location=":memory:")
    embeddings = DeterministicFakeEmbedding(size=VECTOR_SIZE)
    settings = make_settings(embedding_dimensions=VECTOR_SIZE, qdrant_collection="filings")

    documents = [
        Document(
            page_content=f"{company} filing chunk {i} about net sales and segments",
            metadata={"company": company, "chunk_id": i, "source": f"{company}-2025.htm"},
        )
        for company in ("aapl", "tsla")
        for i in range(6)
    ]
    index_documents(documents, client=client, embeddings=embeddings, settings=settings)

    store = build_vector_store(client, embeddings, collection_name="filings")
    return DenseRetriever(store, default_top_k=3)


def test_search_returns_ranked_documents(retriever):
    results = retriever.search("net sales")

    assert len(results) == 3
    assert all(isinstance(d, Document) for d in results)


def test_top_k_overrides_the_default(retriever):
    assert len(retriever.search("net sales", top_k=5)) == 5


def test_results_carry_a_score(retriever):
    """Every stage that reorders results must write its own score, so a printed
    ranking always shows the number that produced it."""
    results = retriever.search("net sales")

    assert all(isinstance(d.metadata["score"], float) for d in results)


def test_scores_are_descending(retriever):
    scores = [d.metadata["score"] for d in retriever.search("net sales", top_k=6)]

    assert scores == sorted(scores, reverse=True)


def test_company_filter_scopes_results(retriever):
    results = retriever.search("net sales", {"company": "tsla"}, top_k=6)

    assert results
    assert {d.metadata["company"] for d in results} == {"tsla"}


def test_no_filter_spans_all_companies(retriever):
    results = retriever.search("net sales", top_k=12)

    assert {d.metadata["company"] for d in results} == {"aapl", "tsla"}


def test_unknown_filter_key_raises(retriever):
    """A silently dropped filter returns another company's chunks and looks
    like a retrieval quality problem rather than a bug."""
    with pytest.raises(RetrievalError, match="Unknown filter key"):
        retriever.search("net sales", {"sector": "tech"})


def test_empty_query_raises(retriever):
    with pytest.raises(RetrievalError, match="empty query"):
        retriever.search("   ")


def test_filter_for_a_company_with_no_data_returns_nothing(retriever):
    """No results is a valid outcome, not an error."""
    assert retriever.search("net sales", {"company": "msft"}) == []


def test_search_fn_adapter_matches_the_harness_signature(retriever):
    search_fn = retriever.as_search_fn()

    results = search_fn("net sales", {"company": "aapl"}, 2)

    assert len(results) == 2
    assert {d.metadata["company"] for d in results} == {"aapl"}


def test_langchain_retriever_adapter_works(retriever):
    lc_retriever = retriever.as_langchain_retriever(filters={"company": "tsla"}, top_k=2)

    results = lc_retriever.invoke("net sales")

    assert len(results) == 2
    assert {d.metadata["company"] for d in results} == {"tsla"}


def test_build_filter_uses_the_nested_payload_path():
    """Bare 'company' would match nothing, silently."""
    qdrant_filter = build_qdrant_filter({"company": "aapl"})

    assert qdrant_filter is not None
    assert isinstance(qdrant_filter.must, list)
    condition = qdrant_filter.must[0]
    assert isinstance(condition, qmodels.FieldCondition)
    assert condition.key == "metadata.company"


def test_build_filter_returns_none_when_unconstrained():
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter({}) is None
