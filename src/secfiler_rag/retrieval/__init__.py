"""Stage 3 — query in, ranked `Document` list out.

Responsibility: every strategy for *finding* context — dense (vector) search
today; sparse (BM25), hybrid fusion and reranking as later modules.

Every strategy exposes the same `(query, filters, top_k) -> list[Document]`
shape, which is what lets the evaluation harness score any of them without
knowing which it is scoring.

This package never imports `indexing`'s write path — only the payload-layout
constants it must agree with.
"""

from secfiler_rag.retrieval.dense import DenseRetriever, build_filter

__all__ = ["DenseRetriever", "build_filter"]
