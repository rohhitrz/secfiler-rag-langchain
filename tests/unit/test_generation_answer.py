"""Generation: the refusal guarantee, citation validation, and chain wiring.

The two tests that matter most are
`test_empty_retrieval_never_reaches_the_model` and
`test_invented_citation_markers_are_dropped`. Both guard properties that a
prompt instruction alone cannot deliver.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_openai import ChatOpenAI

from secfiler_rag.core.exceptions import ConfigurationError, GenerationError
from secfiler_rag.generation.answer import (
    REFUSAL_MESSAGE,
    AnswerGenerator,
    build_llm,
    resolve_citations,
)
from secfiler_rag.generation.context import build_context
from tests.conftest import make_settings


def doc(text="Total net sales were $416,161 million.", company="aapl", chunk_id=12):
    return Document(
        page_content=text,
        metadata={"company": company, "chunk_id": chunk_id, "source": f"{company}-2025.htm"},
    )


class FakeRetriever:
    def __init__(self, documents):
        self._documents = documents
        self.calls = []

    def search(self, query, filters=None, top_k=None):
        self.calls.append((query, dict(filters or {}), top_k))
        return list(self._documents)


class ExplodingModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise RuntimeError("openai is down")


def generator(documents, responses=None, **kwargs):
    return AnswerGenerator(
        FakeRetriever(documents),
        FakeListChatModel(responses=responses or ["Apple reported $416,161 million [1]."]),
        **kwargs,
    )


# --- the refusal guarantee -------------------------------------------------


def test_empty_retrieval_never_reaches_the_model():
    """The guardrail that actually holds.

    A prompt saying "say so if you don't know" is guidance. A model handed an
    empty context and told to answer will still produce something fluent and
    invented. The guarantee is not calling it.
    """
    model = ExplodingModel(responses=["unreachable"])
    gen = AnswerGenerator(FakeRetriever([]), model)

    result = gen.answer("What was Apple's revenue?")

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == ()


def test_refusal_carries_no_fabricated_context():
    result = generator([]).answer("anything")

    assert result.context_blocks == ()
    assert result.retrieved_documents == ()


def test_a_models_refusal_wording_is_detected():
    result = generator(
        [doc()], responses=["The provided filings do not contain this information."]
    ).answer("What is Tesla's dividend?")

    assert result.refused is True


def test_a_normal_answer_is_not_flagged_as_a_refusal():
    result = generator([doc()], responses=["Apple reported $416,161 million [1]."]).answer("q")

    assert result.refused is False


# --- citations -------------------------------------------------------------


def test_citations_resolve_to_the_source_chunk():
    result = generator(
        [doc(company="tsla", chunk_id=21)], responses=["Tesla sells Megapack [1]."]
    ).answer("What does Tesla sell?")

    assert len(result.citations) == 1
    assert result.citations[0].company == "tsla"
    assert result.citations[0].chunk_id == 21
    assert result.citations[0].source == "tsla-2025.htm"


def test_invented_citation_markers_are_dropped():
    """A model citing [7] when 2 blocks were supplied is inventing provenance,
    and an unvalidated citation is worse than none — it looks verifiable."""
    blocks, _ = build_context([doc(chunk_id=1), doc(chunk_id=2)], max_tokens=10_000)

    citations = resolve_citations("Revenue rose [1] and margins improved [7].", blocks)

    assert [c.marker for c in citations] == [1]


def test_repeated_markers_are_deduplicated_in_first_cited_order():
    blocks, _ = build_context([doc(chunk_id=1), doc(chunk_id=2)], max_tokens=10_000)

    citations = resolve_citations("A [2]. B [1]. C [2].", blocks)

    assert [c.marker for c in citations] == [2, 1]


def test_multiple_markers_in_one_group_are_all_resolved():
    blocks, _ = build_context([doc(chunk_id=1), doc(chunk_id=2)], max_tokens=10_000)

    citations = resolve_citations("Both filings agree [1][2].", blocks)

    assert [c.marker for c in citations] == [1, 2]


def test_an_uncited_answer_yields_no_citations():
    citations = resolve_citations("Revenue increased.", [])

    assert citations == ()


# --- wiring ----------------------------------------------------------------


def test_context_blocks_are_returned_for_auditing():
    result = generator([doc(), doc()]).answer("q")

    assert len(result.context_blocks) == 2
    assert len(result.retrieved_documents) == 2


def test_filters_and_top_k_reach_the_retriever():
    retriever = FakeRetriever([doc()])
    gen = AnswerGenerator(retriever, FakeListChatModel(responses=["ok [1]"]), top_k=3)

    gen.answer("q", {"company": "msft"}, top_k=7)

    assert retriever.calls[0][1] == {"company": "msft"}
    assert retriever.calls[0][2] == 7


def test_configured_top_k_is_the_default():
    retriever = FakeRetriever([doc()])

    AnswerGenerator(retriever, FakeListChatModel(responses=["ok"]), top_k=4).answer("q")

    assert retriever.calls[0][2] == 4


def test_context_budget_limits_what_the_model_sees():
    big = doc(text="word " * 400)
    gen = generator([big, big, big], max_context_tokens=600)

    result = gen.answer("q")

    assert 0 < len(result.context_blocks) < 3


def test_empty_question_raises():
    with pytest.raises(GenerationError, match="empty question"):
        generator([doc()]).answer("   ")


def test_model_failure_is_wrapped_in_our_error_type():
    gen = AnswerGenerator(FakeRetriever([doc()]), ExplodingModel(responses=["x"]))

    with pytest.raises(GenerationError, match="Answer generation failed"):
        gen.answer("q")


def test_missing_api_key_raises_a_named_configuration_error(clean_env):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_llm(make_settings(openai_api_key=None))


def test_llm_is_built_from_settings(clean_env):
    llm = build_llm(make_settings(openai_api_key="sk-test", llm_model="gpt-4o-mini"))

    # Narrowing is the assertion: build_llm must return a configured OpenAI
    # chat model, not merely "something invokable".
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.0
