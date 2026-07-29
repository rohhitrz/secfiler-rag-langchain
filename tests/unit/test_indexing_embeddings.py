"""Embedding construction: fail fast and loudly on a missing credential."""

import pytest
from langchain_openai import OpenAIEmbeddings

from secfiler_rag.core.exceptions import ConfigurationError
from secfiler_rag.indexing.embeddings import build_embeddings
from tests.conftest import make_settings as _settings


def test_missing_api_key_raises_a_named_configuration_error(clean_env):
    """Better than a 401 surfacing from three frames inside the OpenAI client."""
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_embeddings(_settings(openai_api_key=None))


def test_model_is_built_from_settings(clean_env):
    embeddings = build_embeddings(
        _settings(
            openai_api_key="sk-test",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=256,
            embedding_batch_size=25,
        )
    )

    # Narrowing from the Embeddings interface is the assertion: build_embeddings
    # must hand back a configured OpenAI model, not just "something embeddable".
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-large"
    assert embeddings.dimensions == 256
    assert embeddings.chunk_size == 25


def test_no_network_call_happens_at_construction(clean_env):
    """Construction must stay free — importing or wiring cannot hit the API."""
    build_embeddings(_settings(openai_api_key="sk-not-a-real-key"))
