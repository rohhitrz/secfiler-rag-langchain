"""Stage 3 — query in, ranked `Document` list out.

Responsibility: every strategy for *finding* context.

| Strategy | Mechanism | Strength |
|---|---|---|
| `DenseRetriever` | Embeddings + Qdrant ANN | Paraphrase, synonyms, intent |
| `SparseRetriever` | BM25 over an in-memory corpus | Rare exact terms, identifiers |
| `HybridRetriever` | Both, fused by RRF | Neither failure mode alone |
| `RerankingRetriever` | Cross-encoder over a shortlist | Reads query and document together |

Every strategy exposes the same `(query, filters, top_k) -> list[Document]`
shape, which is what lets the evaluation harness score any of them without
knowing which it is scoring. The shared filter vocabulary lives in `filters`,
so a key one strategy honours cannot be one another silently ignores.

Every stage that reorders results overwrites `metadata["score"]` with its own
number, so a printed ranking always shows the score that explains its order.
"""

from secfiler_rag.retrieval.dense import DenseRetriever
from secfiler_rag.retrieval.filters import build_qdrant_filter, matches, validate_filters
from secfiler_rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from secfiler_rag.retrieval.hybrid import HybridRetriever, Retriever
from secfiler_rag.retrieval.rerank import RerankClient, RerankingRetriever, build_rerank_client
from secfiler_rag.retrieval.sparse import SparseRetriever, tokenize

__all__ = [
    "DEFAULT_RRF_K",
    "DenseRetriever",
    "HybridRetriever",
    "RerankClient",
    "RerankingRetriever",
    "Retriever",
    "SparseRetriever",
    "build_qdrant_filter",
    "build_rerank_client",
    "matches",
    "reciprocal_rank_fusion",
    "tokenize",
    "validate_filters",
]
