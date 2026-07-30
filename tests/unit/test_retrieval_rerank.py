"""Reranking: index alignment, score mapping, graceful degradation.

The critical test is `test_scores_map_by_returned_index_not_response_order`.
Mispairing documents with scores produces output that still *looks* like a
plausible ranking, so nothing downstream would ever notice.
"""

from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from secfiler_rag.core.exceptions import ConfigurationError, RetrievalError
from secfiler_rag.retrieval.rerank import RerankingRetriever, build_rerank_client
from tests.conftest import make_settings


@dataclass
class FakeResult:
    index: int
    relevance_score: float


@dataclass
class FakeResponse:
    results: list[FakeResult]


class FakeRerankClient:
    """Returns a scripted ranking and records what it was asked."""

    def __init__(self, ranking, *, shuffle_response=False):
        self._ranking = ranking
        self._shuffle_response = shuffle_response
        self.calls = []

    def rerank(self, *, model, query, documents, top_n):
        self.calls.append({"model": model, "query": query, "documents": documents, "top_n": top_n})
        results = [FakeResult(index=i, relevance_score=s) for i, s in self._ranking]
        if self._shuffle_response:
            # Deliberately NOT sorted by score: the provider's contract is the
            # `index` field, not the order results happen to arrive in.
            results = list(reversed(results))
        return FakeResponse(results=results)


class ExplodingClient:
    """Always fails, with a non-retryable error by default."""

    def __init__(self, error=None, fail_times=None):
        self._error = error or RuntimeError("cohere is down")
        self._fail_times = fail_times
        self.calls = 0

    def rerank(self, **kwargs):
        self.calls += 1
        if self._fail_times is not None and self.calls > self._fail_times:
            return FakeResponse(results=[FakeResult(index=0, relevance_score=0.9)])
        raise self._error


class RateLimitedError(Exception):
    """Stands in for cohere.TooManyRequestsError without importing it."""

    def __init__(self):
        super().__init__("status_code: 429, rate limit exceeded")

    def __class_getitem__(cls, _):  # pragma: no cover - unused
        return cls


# The class name is what _is_retryable matches on, alongside the message.
RateLimitedError.__name__ = "TooManyRequestsError"


class FakeBase:
    """A base retriever returning documents in a fixed order."""

    def __init__(self, chunk_ids):
        self._chunk_ids = chunk_ids
        self.calls = []

    def search(self, query, filters=None, top_k=None):
        self.calls.append((query, dict(filters or {}), top_k))
        ids = self._chunk_ids[: top_k or len(self._chunk_ids)]
        return [
            Document(
                page_content=f"chunk {i} text",
                metadata={"company": "aapl", "chunk_id": i, "score": 0.5},
            )
            for i in ids
        ]


def ids(documents):
    return [d.metadata["chunk_id"] for d in documents]


def test_reranker_reorders_the_base_ranking():
    base = FakeBase([10, 20, 30])
    # The cross-encoder disagrees with the retriever: input position 2 is best.
    client = FakeRerankClient([(2, 0.99), (0, 0.40), (1, 0.10)])

    results = RerankingRetriever(base, client, candidate_k=3, default_top_k=3).search("q")

    assert ids(results) == [30, 10, 20]


def test_scores_map_by_returned_index_not_response_order():
    """The discipline this module exists to enforce.

    The fake returns results in reverse-score order. Code that pairs documents
    with scores by response position would mispair every one — and the output
    would still look like a plausible ranking.
    """
    base = FakeBase([10, 20, 30])
    client = FakeRerankClient([(2, 0.99), (0, 0.40), (1, 0.10)], shuffle_response=True)

    results = RerankingRetriever(base, client, candidate_k=3, default_top_k=3).search("q")

    by_chunk = {d.metadata["chunk_id"]: d.metadata["score"] for d in results}
    assert by_chunk == {30: pytest.approx(0.99), 10: pytest.approx(0.40), 20: pytest.approx(0.10)}


def test_rerank_score_overwrites_the_retriever_score():
    base = FakeBase([10])
    client = FakeRerankClient([(0, 0.91)])

    results = RerankingRetriever(base, client, candidate_k=1, default_top_k=1).search("q")

    assert results[0].metadata["score"] == pytest.approx(0.91)


def test_pre_rerank_rank_is_recorded_for_auditing():
    base = FakeBase([10, 20, 30])
    client = FakeRerankClient([(2, 0.99), (0, 0.40)])

    results = RerankingRetriever(base, client, candidate_k=3, default_top_k=2).search("q")

    assert results[0].metadata["rank_before_rerank"] == 3
    assert results[1].metadata["rank_before_rerank"] == 1


def test_candidate_pool_is_wider_than_the_returned_set():
    """Reranking reorders; it cannot retrieve. A chunk outside the pool is lost."""
    base = FakeBase(list(range(20)))
    client = FakeRerankClient([(0, 0.9)])

    RerankingRetriever(base, client, candidate_k=10, default_top_k=3).search("q")

    assert base.calls[0][2] == 10


def test_candidate_pool_never_narrower_than_the_requested_top_k():
    base = FakeBase(list(range(20)))
    client = FakeRerankClient([(0, 0.9)])

    RerankingRetriever(base, client, candidate_k=3).search("q", top_k=8)

    assert base.calls[0][2] == 8


def test_only_top_k_is_returned():
    base = FakeBase([10, 20, 30, 40])
    client = FakeRerankClient([(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)])

    results = RerankingRetriever(base, client, candidate_k=4, default_top_k=2).search("q")

    assert len(results) == 2


def test_top_n_requested_never_exceeds_the_candidate_count():
    base = FakeBase([10, 20])
    client = FakeRerankClient([(0, 0.9), (1, 0.8)])

    RerankingRetriever(base, client, candidate_k=10, default_top_k=5).search("q")

    assert client.calls[0]["top_n"] == 2


def test_filters_are_forwarded_to_the_base_retriever():
    base = FakeBase([10])
    client = FakeRerankClient([(0, 0.9)])

    RerankingRetriever(base, client).search("q", {"company": "tsla"})

    assert base.calls[0][1] == {"company": "tsla"}


def test_reranker_failure_degrades_to_the_base_ranking():
    """Reranking improves ordering; the system is correct without it."""
    base = FakeBase([10, 20, 30])

    results = RerankingRetriever(base, ExplodingClient(), candidate_k=3, default_top_k=2).search(
        "q"
    )

    assert ids(results) == [10, 20]


def test_failure_can_be_made_strict():
    base = FakeBase([10])

    with pytest.raises(RetrievalError, match="Reranking failed"):
        RerankingRetriever(base, ExplodingClient(), fail_open=False).search("q")


def test_no_candidates_skips_the_reranker_entirely():
    """Never spend an API call on an empty list."""
    client = FakeRerankClient([])

    results = RerankingRetriever(FakeBase([]), client).search("q")

    assert results == []
    assert client.calls == []


def test_out_of_range_index_from_the_provider_is_skipped():
    base = FakeBase([10, 20])
    client = FakeRerankClient([(99, 0.99), (0, 0.5)])

    results = RerankingRetriever(base, client, candidate_k=2, default_top_k=2).search("q")

    assert ids(results) == [10]


def test_configured_model_is_passed_through():
    base = FakeBase([10])
    client = FakeRerankClient([(0, 0.9)])

    RerankingRetriever(base, client, model="rerank-v3.5").search("q")

    assert client.calls[0]["model"] == "rerank-v3.5"


def test_search_fn_adapter_matches_the_harness_signature():
    base = FakeBase([10, 20])
    client = FakeRerankClient([(1, 0.9), (0, 0.8)])

    results = RerankingRetriever(base, client, candidate_k=2).as_search_fn()("q", {}, 2)

    assert ids(results) == [20, 10]


def test_missing_api_key_raises_a_named_configuration_error(clean_env):
    with pytest.raises(ConfigurationError, match="COHERE_API_KEY"):
        build_rerank_client(make_settings(cohere_api_key=None))


def test_rerank_defaults_are_the_measured_funnel_widths(clean_env):
    settings = make_settings()

    assert settings.rerank_candidate_k == 10
    assert settings.rerank_top_k == 3


# --- retry behaviour -------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds():
    """Cohere's trial tier allows ~10 calls/minute and an eval issues dozens."""
    base = FakeBase([10])
    client = ExplodingClient(error=RateLimitedError(), fail_times=2)
    slept: list[float] = []

    results = RerankingRetriever(
        base, client, candidate_k=1, default_top_k=1, sleep=slept.append
    ).search("q")

    assert client.calls == 3
    assert ids(results) == [10]


def test_backoff_doubles_between_attempts():
    base = FakeBase([10])
    client = ExplodingClient(error=RateLimitedError())
    slept: list[float] = []

    RerankingRetriever(base, client, max_retries=3, backoff_seconds=2.0, sleep=slept.append).search(
        "q"
    )

    assert slept == [2.0, 4.0, 8.0]


def test_retries_are_bounded():
    base = FakeBase([10])
    client = ExplodingClient(error=RateLimitedError())

    RerankingRetriever(base, client, max_retries=2, sleep=lambda _: None).search("q")

    assert client.calls == 3  # the first attempt plus two retries


def test_non_retryable_errors_are_not_retried():
    """Retrying an auth failure just adds latency to a guaranteed failure."""
    base = FakeBase([10])
    client = ExplodingClient(error=RuntimeError("invalid api key"))

    RerankingRetriever(base, client, max_retries=3, sleep=lambda _: None).search("q")

    assert client.calls == 1


def test_exhausted_retries_still_degrade_when_failing_open():
    base = FakeBase([10, 20])
    client = ExplodingClient(error=RateLimitedError())

    results = RerankingRetriever(
        base, client, candidate_k=2, default_top_k=2, max_retries=1, sleep=lambda _: None
    ).search("q")

    assert ids(results) == [10, 20]


def test_strict_mode_surfaces_rate_limits_instead_of_hiding_them():
    """The eval CLI uses fail_open=False for exactly this reason: a silently
    absent reranker gets measured as if it ran."""
    base = FakeBase([10])
    client = ExplodingClient(error=RateLimitedError())

    with pytest.raises(RetrievalError, match="Reranking failed"):
        RerankingRetriever(
            base, client, fail_open=False, max_retries=1, sleep=lambda _: None
        ).search("q")
