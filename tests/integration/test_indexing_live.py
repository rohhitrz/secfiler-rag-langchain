"""Indexing against live Qdrant and the real OpenAI embedding API.

Everything the unit tests cannot prove: that the configured model really
returns 1536-dimensional vectors, that a real Qdrant server accepts our
collection config and payload index, and that semantic search over real filing
text returns the right company's chunks.

Requires `docker compose up -d` and `OPENAI_API_KEY`. Skipped otherwise.

    uv run pytest -m integration
"""

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from secfiler_rag.config.settings import Settings
from secfiler_rag.indexing import (
    COMPANY_PAYLOAD_FIELD,
    build_client,
    build_embeddings,
    build_vector_store,
    count_points,
    index_documents,
)
from secfiler_rag.ingestion import ingest_all

pytestmark = pytest.mark.integration

TEST_COLLECTION = "filings_integration_test"


def _settings() -> Settings:
    return Settings(qdrant_collection=TEST_COLLECTION)


def _qdrant_available(settings: Settings) -> bool:
    try:
        build_client(settings).get_collections()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module")
def settings() -> Settings:
    settings = _settings()
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY not set")
    if not _qdrant_available(settings):
        pytest.skip("Qdrant not reachable — run `docker compose up -d`")
    return settings


@pytest.fixture(scope="module")
def client(settings: Settings) -> Iterator[QdrantClient]:
    qdrant = build_client(settings)
    yield qdrant
    if qdrant.collection_exists(TEST_COLLECTION):
        qdrant.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="module")
def indexed(client: QdrantClient, settings: Settings) -> int:
    """Index a small real slice: the first 20 chunks of each company."""
    documents = ingest_all()
    subset = [d for d in documents if d.metadata["chunk_id"] < 20]
    return index_documents(
        subset,
        client=client,
        embeddings=build_embeddings(settings),
        settings=settings,
        recreate=True,
    )


def test_real_embeddings_match_the_configured_dimension(settings: Settings) -> None:
    vector = build_embeddings(settings).embed_query("total net sales")

    assert len(vector) == settings.embedding_dimensions


def test_points_land_in_the_collection(client: QdrantClient, indexed: int) -> None:
    assert count_points(client, TEST_COLLECTION) == indexed


def test_reindexing_is_idempotent_against_a_live_server(
    client: QdrantClient, settings: Settings, indexed: int
) -> None:
    documents = [d for d in ingest_all() if d.metadata["chunk_id"] < 20]

    index_documents(
        documents, client=client, embeddings=build_embeddings(settings), settings=settings
    )

    assert count_points(client, TEST_COLLECTION) == indexed


def test_semantic_search_returns_company_scoped_chunks(
    client: QdrantClient, settings: Settings, indexed: int
) -> None:
    store = build_vector_store(client, build_embeddings(settings), collection_name=TEST_COLLECTION)

    results = store.similarity_search(
        "annual report cover page and fiscal year",
        k=5,
        filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=COMPANY_PAYLOAD_FIELD, match=qmodels.MatchValue(value="tsla")
                )
            ]
        ),
    )

    assert results
    assert {d.metadata["company"] for d in results} == {"tsla"}
