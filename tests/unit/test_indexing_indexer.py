"""Indexing behaviour: idempotency, metadata round-trip, and filterability.

Uses in-memory Qdrant plus a deterministic fake embedder, so these exercise the
real write path — real upserts, real payloads, real filters — without an API
key or a container.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from secfiler_rag.core.exceptions import IndexingError
from secfiler_rag.indexing.collection import COMPANY_PAYLOAD_FIELD, count_points
from secfiler_rag.indexing.indexer import build_vector_store, index_documents
from tests.conftest import make_settings

VECTOR_SIZE = 32


@pytest.fixture
def client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=VECTOR_SIZE)


@pytest.fixture
def settings():
    return make_settings(
        embedding_dimensions=VECTOR_SIZE,
        embedding_batch_size=2,  # small, so batching is actually exercised
        qdrant_collection="filings",
    )


def docs(company="aapl", count=5):
    return [
        Document(
            page_content=f"{company} chunk {i} discussing net sales and segment revenue.",
            metadata={"company": company, "chunk_id": i, "source": f"{company}-2025.htm"},
        )
        for i in range(count)
    ]


def _index(client, embeddings, settings, documents, **kwargs):
    return index_documents(
        documents, client=client, embeddings=embeddings, settings=settings, **kwargs
    )


def test_documents_are_written_as_points(client, embeddings, settings):
    written = _index(client, embeddings, settings, docs(count=5))

    assert written == 5
    assert count_points(client, "filings") == 5


def test_reindexing_overwrites_instead_of_duplicating(client, embeddings, settings):
    """The whole point of deterministic IDs.

    With auto-generated IDs this second run would double the corpus, and the
    symptom would not be an error — just duplicates crowding every top-k.
    """
    _index(client, embeddings, settings, docs(count=5))
    _index(client, embeddings, settings, docs(count=5))

    assert count_points(client, "filings") == 5


def test_reindexing_updates_changed_content(client, embeddings, settings):
    _index(client, embeddings, settings, docs(count=1))

    changed = [
        Document(
            page_content="rewritten content after a chunker change",
            metadata={"company": "aapl", "chunk_id": 0, "source": "aapl-2025.htm"},
        )
    ]
    _index(client, embeddings, settings, changed)

    stored = client.scroll(collection_name="filings", limit=1)[0][0]
    assert stored.payload is not None
    assert stored.payload["page_content"] == "rewritten content after a chunker change"


def test_companies_do_not_overwrite_each_other(client, embeddings, settings):
    """chunk_id restarts per company; identity is the pair."""
    _index(client, embeddings, settings, docs("aapl", 3))
    _index(client, embeddings, settings, docs("tsla", 3))

    assert count_points(client, "filings") == 6


def test_metadata_survives_the_round_trip(client, embeddings, settings):
    _index(client, embeddings, settings, docs(count=1))

    payload = client.scroll(collection_name="filings", limit=1)[0][0].payload
    assert payload is not None
    assert payload["metadata"]["company"] == "aapl"
    assert payload["metadata"]["chunk_id"] == 0
    assert payload["metadata"]["source"] == "aapl-2025.htm"


def test_company_filter_scopes_results(client, embeddings, settings):
    _index(client, embeddings, settings, docs("aapl", 4))
    _index(client, embeddings, settings, docs("tsla", 4))
    store = build_vector_store(client, embeddings, collection_name="filings")

    results = store.similarity_search(
        "net sales",
        k=10,
        filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=COMPANY_PAYLOAD_FIELD,
                    match=qmodels.MatchValue(value="tsla"),
                )
            ]
        ),
    )

    assert results
    assert {d.metadata["company"] for d in results} == {"tsla"}


def test_search_returns_documents_with_metadata(client, embeddings, settings):
    _index(client, embeddings, settings, docs(count=5))
    store = build_vector_store(client, embeddings, collection_name="filings")

    results = store.similarity_search("net sales", k=3)

    assert len(results) == 3
    for doc in results:
        assert doc.metadata["company"] == "aapl"
        assert isinstance(doc.metadata["chunk_id"], int)


def test_batching_writes_every_document(client, embeddings, settings):
    """batch_size is 2 here, so 7 documents span four batches."""
    written = _index(client, embeddings, settings, docs(count=7))

    assert written == 7
    assert count_points(client, "filings") == 7


def test_recreate_clears_previous_points(client, embeddings, settings):
    _index(client, embeddings, settings, docs("aapl", 5))

    _index(client, embeddings, settings, docs("tsla", 2), recreate=True)

    assert count_points(client, "filings") == 2


def test_empty_document_list_raises(client, embeddings, settings):
    with pytest.raises(IndexingError, match="No documents"):
        _index(client, embeddings, settings, [])


def test_duplicate_chunk_identity_raises(client, embeddings, settings):
    duplicated = docs(count=1) * 2

    with pytest.raises(IndexingError, match="Duplicate point IDs"):
        _index(client, embeddings, settings, duplicated)


def test_dimension_mismatch_with_existing_collection_raises(client, embeddings, settings):
    _index(client, embeddings, settings, docs(count=1))
    wider = settings.model_copy(update={"embedding_dimensions": VECTOR_SIZE * 2})

    with pytest.raises(IndexingError, match="vector size"):
        _index(client, embeddings, wider, docs(count=1))
