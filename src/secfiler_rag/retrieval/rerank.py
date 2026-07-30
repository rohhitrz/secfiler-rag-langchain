"""Cross-encoder reranking — the last stage before context reaches the LLM.

**Why a cross-encoder is categorically different from everything upstream.**

Dense retrieval compares two vectors that were computed *independently*: the
query never saw the document, and the document never saw the query. That is
what makes it fast — every chunk is embedded once, offline — and it is also its
ceiling. A bi-encoder must compress a whole chunk into one vector before
knowing what will be asked of it.

A cross-encoder reads the query and the document **together** in a single
forward pass and scores that pair directly. It cannot be precomputed, so it is
far too expensive to run over a corpus — and it is markedly more accurate over
a shortlist. That is the entire architecture of the funnel: cheap-and-broad
retrievers narrow 1,309 chunks to ~10, then an expensive-and-precise model
orders those 10.

It is also the answer to a failure fusion provably cannot fix. When dense ranks
the answer 6th and sparse ranks it 5th, both retrievers *agree* it is mediocre,
and RRF rewards agreement — no way of recombining those rankings promotes it.
Only a model that actually reads the pair can.

**Index alignment is the discipline that matters.** Cohere returns
`results[i].index` — a position into the list you sent, not your own ID. The
results happen to arrive sorted by relevance, but the code must never rely on
that: mapping by position in the *response* rather than by the returned
`index` silently scrambles document-to-score pairing, and the output still
looks like a plausible ranking.

**Reranking degrades, it does not fail — except when measuring.** In a service,
a reranker outage should return the un-reranked candidates with a warning
rather than failing the query: reranking improves ordering, and the system is
still correct without it.

In an *evaluation* that same behaviour is a trap. Cohere's trial tier allows
about ten requests per minute, and an eval run issues dozens back to back — so
the harness silently measured the fallback ranking and would have reported it
as the reranker's score. `fail_open=False` exists for exactly that reason, and
the eval CLI uses it. **You cannot measure a component that is quietly not
running.**

Rate limits are handled first by bounded retry with exponential backoff, which
is what a production caller of any per-request paid API needs regardless.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from secfiler_rag.config import Settings, get_settings
from secfiler_rag.core.exceptions import ConfigurationError, RetrievalError
from secfiler_rag.core.logging import get_logger
from secfiler_rag.retrieval.hybrid import Retriever

log = get_logger(__name__)


class RerankResult(Protocol):
    """One scored candidate: a position into the input list, plus its score."""

    # Read-only members. A mutable protocol attribute is invariant, which no
    # provider's concrete response model would satisfy structurally.
    @property
    def index(self) -> int:
        """Position of this document in the list that was sent."""

    @property
    def relevance_score(self) -> float:
        """Cross-encoder relevance for the (query, document) pair."""


class RerankResponse(Protocol):
    """A rerank response."""

    @property
    def results(self) -> Sequence[RerankResult]:
        """Scored candidates, in provider-defined order."""


class RerankClient(Protocol):
    """The single call this module needs.

    A Protocol rather than the concrete Cohere client, so tests inject a fake
    and a different provider becomes a swap rather than a rewrite.
    """

    def rerank(
        self, *, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> RerankResponse:
        """Score documents against the query."""
        ...


class RerankingRetriever:
    """Wraps any retriever and reorders its candidates with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        client: RerankClient,
        *,
        model: str = "rerank-v3.5",
        candidate_k: int = 10,
        default_top_k: int = 3,
        fail_open: bool = True,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the rerank stage.

        Args:
            base: Any retriever — dense, sparse or hybrid. Reranking composes
                with all of them, which is what lets the harness measure
                rerank-over-dense against rerank-over-hybrid.
            client: Anything satisfying `RerankClient`.
            model: Cohere rerank model.
            candidate_k: Candidates requested from `base`. **A chunk outside
                this pool can never be recovered** — reranking reorders, it
                does not retrieve.
            default_top_k: Chunks kept when a caller does not specify.
            fail_open: On reranker failure, return the base ranking with a
                warning instead of raising. Correct for a service; **wrong for
                an evaluation**, which must not silently score the fallback.
            max_retries: Attempts after the first for a retryable failure.
                Five with a 2s base gives ~62s of cumulative backoff, enough to
                clear a ten-per-minute rate limit.
            backoff_seconds: Base delay, doubled per attempt.
            sleep: Injectable for tests, so retry logic costs no wall time.
        """
        self._base = base
        self._client = client
        self._model = model
        self._candidate_k = candidate_k
        self._default_top_k = default_top_k
        self._fail_open = fail_open
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve candidates, rerank them, and return the best.

        Returns:
            Documents ordered by cross-encoder relevance. Each carries `score`
            (the rerank score) and `rank_before_rerank`, so an audit can show
            whether reranking actually moved anything.
        """
        k = top_k if top_k is not None else self._default_top_k
        # Never ask for fewer candidates than we intend to return, and never
        # narrow the pool below the configured width just because k is small.
        candidate_k = max(self._candidate_k, k)

        candidates = list(self._base.search(query, filters, candidate_k))
        if not candidates:
            return []

        try:
            response = self._rerank_with_retry(query, candidates, k)
        except Exception as exc:
            if not self._fail_open:
                # Translate the provider's exception into ours, so a caller can
                # catch SecfilerRagError and be certain it covers every failure
                # this package raises.
                raise RetrievalError(f"Reranking failed: {exc}") from exc
            # Reranking is an enhancement, not a dependency. Losing it costs
            # ordering quality; failing the query costs the answer.
            log.warning(
                "reranker unavailable, falling back to base ranking",
                extra={"error": str(exc), "query": query},
            )
            return candidates[:k]

        return self._apply(response, candidates, k)

    def _rerank_with_retry(self, query: str, candidates: list[Document], k: int) -> RerankResponse:
        """Call the reranker, retrying transient failures with backoff.

        Rate limiting is the expected failure for any per-request paid API —
        Cohere's trial tier allows roughly ten calls per minute, and an eval
        run issues dozens. Retrying is what makes the difference between a
        measurable component and a silently absent one.
        """
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.rerank(
                    model=self._model,
                    query=query,
                    documents=[doc.page_content for doc in candidates],
                    top_n=min(k, len(candidates)),
                )
            except Exception as exc:
                if attempt == self._max_retries or not _is_retryable(exc):
                    raise
                delay = self._backoff_seconds * (2**attempt)
                log.warning(
                    "reranker call failed, retrying",
                    extra={"attempt": attempt + 1, "delay_s": delay, "error": str(exc)[:200]},
                )
                self._sleep(delay)

        # Unreachable: the final attempt either returns or re-raises.
        raise AssertionError("retry loop exited without returning")  # pragma: no cover

    def _apply(
        self, response: RerankResponse, candidates: list[Document], k: int
    ) -> list[Document]:
        """Map scores back onto the documents that produced them.

        The mapping goes through `result.index` — a position into the list we
        sent — never through the result's position in the response. Cohere does
        return results sorted by relevance, but relying on that would silently
        mispair documents and scores if it ever stopped being true, and the
        output would still look like a plausible ranking.
        """
        reranked = []
        for result in response.results:
            if not 0 <= result.index < len(candidates):  # pragma: no cover - provider bug
                log.warning(
                    "reranker returned an out-of-range index", extra={"index": result.index}
                )
                continue

            document = candidates[result.index]
            reranked.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "score": float(result.relevance_score),
                        # 1-based position before reranking, so an audit can
                        # show what the cross-encoder actually changed.
                        "rank_before_rerank": result.index + 1,
                    },
                )
            )

        log.debug(
            "reranked candidates",
            extra={
                "candidates": len(candidates),
                "returned": len(reranked),
                "moved": sum(
                    1
                    for i, d in enumerate(reranked, start=1)
                    if d.metadata["rank_before_rerank"] != i
                ),
            },
        )
        return reranked[:k]

    def as_search_fn(self) -> Any:
        """Adapt to the eval harness's `(query, filters, top_k)` signature."""

        def search_fn(query: str, filters: Mapping[str, Any], top_k: int) -> Sequence[Document]:
            return self.search(query, filters, top_k)

        return search_fn


def _is_retryable(exc: Exception) -> bool:
    """Whether a failure is worth retrying.

    Matched on the exception's name and message rather than on a provider
    exception class, so this module keeps its Protocol-only coupling to the
    reranker and a different provider needs no change here.
    """
    signature = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in signature
        for marker in ("toomanyrequests", "rate limit", "ratelimit", "429", "timeout", "503")
    )


def build_rerank_client(settings: Settings | None = None) -> RerankClient:
    """Build the Cohere client.

    Constructed on call rather than at import, so the package imports without
    credentials — the same rule as the embedding client.

    Raises:
        ConfigurationError: If `COHERE_API_KEY` is not set.
    """
    settings = settings or get_settings()

    if settings.cohere_api_key is None:
        raise ConfigurationError(
            "COHERE_API_KEY is not set — required for reranking. "
            "Add it to your .env file (see .env.example), or run without reranking."
        )

    import cohere

    client: RerankClient = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
    return client
