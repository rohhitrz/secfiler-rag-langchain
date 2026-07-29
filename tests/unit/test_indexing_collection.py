"""Collection lifecycle against a real Qdrant engine, in memory.

`QdrantClient(location=":memory:")` runs the actual local engine — real
collection config, real filters, real payload indexes — with no Docker and no
network. That makes these genuine behaviour tests rather than mock theatre,
while still running in milliseconds.
"""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from secfiler_rag.core.exceptions import IndexingError
from secfiler_rag.indexing.collection import (
    COMPANY_PAYLOAD_FIELD,
    count_points,
    ensure_collection,
    ensure_payload_index,
)


@pytest.fixture
def client():
    return QdrantClient(location=":memory:")


def test_collection_is_created_when_absent(client):
    created = ensure_collection(client, collection_name="filings", vector_size=8)

    assert created is True
    assert client.collection_exists("filings")


def test_second_call_verifies_instead_of_recreating(client):
    ensure_collection(client, collection_name="filings", vector_size=8)

    assert ensure_collection(client, collection_name="filings", vector_size=8) is False


def test_vector_size_mismatch_fails_loudly(client):
    """The failure that is otherwise invisible until retrieval returns nonsense."""
    ensure_collection(client, collection_name="filings", vector_size=8)

    with pytest.raises(IndexingError, match="vector size 8"):
        ensure_collection(client, collection_name="filings", vector_size=1536)


def test_recreate_drops_existing_data(client):
    ensure_collection(client, collection_name="filings", vector_size=8)
    client.upsert(
        collection_name="filings",
        points=[qmodels.PointStruct(id=1, vector=[0.1] * 8, payload={})],
    )
    assert count_points(client, "filings") == 1

    ensure_collection(client, collection_name="filings", vector_size=8, recreate=True)

    assert count_points(client, "filings") == 0


def test_recreate_allows_changing_vector_size(client):
    ensure_collection(client, collection_name="filings", vector_size=8)

    created = ensure_collection(client, collection_name="filings", vector_size=16, recreate=True)

    assert created is True


def test_cosine_distance_is_configured(client):
    ensure_collection(client, collection_name="filings", vector_size=8)

    params = client.get_collection("filings").config.params.vectors

    assert params.distance.lower() == "cosine"


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
def test_payload_index_is_idempotent(client):
    """Local mode ignores payload indexes, so this proves only that repeated
    calls are safe. That the index actually narrows the search is covered by
    the live-server integration test — a limitation worth naming rather than
    papering over."""
    ensure_collection(client, collection_name="filings", vector_size=8)

    ensure_payload_index(client, collection_name="filings")
    ensure_payload_index(client, collection_name="filings")  # must not raise


def test_company_field_addresses_the_nested_payload():
    """LangChain nests metadata, so a bare 'company' filter would match nothing."""
    assert COMPANY_PAYLOAD_FIELD == "metadata.company"
