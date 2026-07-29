"""Sparse (BM25) retrieval — lexical matching over the chunk corpus.

**Why keep a lexical retriever at all when you have embeddings?** Because they
fail in opposite directions. Dense search generalises: it finds "net sales"
when you ask about "revenue", and it also happily returns eleven chunks that
are *all* about derivative instruments in an essentially arbitrary order. BM25
does not generalise at all — but it matches rare, exact terms decisively.
`Megapack`, `Powerwall`, `Item 1A`, a specific dollar figure: these are tokens,
not concepts, and an embedding blurs them into their neighbourhood.

**What BM25 actually computes.** For each query term, a document scores higher
when the term appears often in it (term frequency, with saturation so the
tenth occurrence adds little) and when the term is rare across the corpus
(inverse document frequency), adjusted for document length so long chunks are
not rewarded for size alone. No training, no vectors, no API call.

**Tokenizer symmetry is mandatory.** The *same* tokenizer must process the
corpus at index time and the query at search time. Lowercase one side only and
`Megapack` never matches `megapack`; strip punctuation differently and the
retriever silently finds less. This is why there is exactly one `tokenize()`
here and both paths call it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.core.logging import get_logger
from secfiler_rag.retrieval.filters import matches

log = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    The single tokenizer used for **both** corpus indexing and queries.
    Deliberately crude: no stemming, no stop-word removal. Stemming would help
    `hedges` match `hedging` but would also collapse distinct financial terms,
    and BM25's IDF already discounts words that appear everywhere — which is
    what stop-word lists are usually for.

    Note what this discards: `$416,161` becomes `416` and `161`. Acceptable
    because the surrounding label carries the meaning, and because splitting on
    punctuation consistently matters more than splitting it well.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class SparseRetriever:
    """BM25 search over an in-memory corpus of chunks.

    The index is built once at construction and held in process memory. That is
    correct for a batch script and for a service that builds it at startup —
    and wrong for anything that rebuilds it per request, which is the classic
    "it was fast in the eval script" regression.
    """

    def __init__(self, documents: Sequence[Document], *, default_top_k: int = 5) -> None:
        if not documents:
            raise RetrievalError("Cannot build a BM25 index from an empty corpus")

        self._documents = list(documents)
        self._default_top_k = default_top_k
        self._bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self._documents])

        log.debug("built bm25 index", extra={"documents": len(self._documents)})

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve the highest-scoring chunks for a query.

        Args:
            query: Natural-language or keyword query.
            filters: Metadata constraints applied after scoring.
            top_k: Number of chunks to return.

        Returns:
            Documents ranked by BM25 score, each carrying `score` in metadata.

        Raises:
            RetrievalError: If the query is empty or a filter key is unknown.
        """
        if not query.strip():
            raise RetrievalError("Cannot retrieve for an empty query")

        k = top_k if top_k is not None else self._default_top_k
        scores = self._bm25.get_scores(tokenize(query))

        # Filtering happens after scoring, so IDF is computed over the whole
        # corpus rather than per company. That keeps scores comparable across
        # an unfiltered query and a filtered one — see the class docstring
        # trade-off note in ADR 0011.
        candidates = [
            (score, index)
            for index, score in enumerate(scores)
            if score > 0 and matches(self._documents[index].metadata, filters)
        ]
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))

        results = []
        for score, index in candidates[:k]:
            document = self._documents[index]
            results.append(
                Document(
                    page_content=document.page_content,
                    metadata={**document.metadata, "score": float(score)},
                )
            )

        log.debug(
            "sparse search",
            extra={"query": query, "filters": dict(filters or {}), "results": len(results)},
        )
        return results

    def as_search_fn(self) -> Any:
        """Adapt to the eval harness's `(query, filters, top_k)` signature."""

        def search_fn(query: str, filters: Mapping[str, Any], top_k: int) -> Sequence[Document]:
            return self.search(query, filters, top_k)

        return search_fn
