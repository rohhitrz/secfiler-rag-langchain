"""Stage 3 — query in, ranked `Document` list out.

Responsibility: every strategy for *finding* context — dense (vector), sparse
(BM25), hybrid fusion (RRF), metadata filtering, and cross-encoder reranking.

Every retriever exposes the same LangChain `BaseRetriever` interface, which is
what lets the evaluation harness stay strategy-agnostic: it takes a retriever,
not a hard-coded search function.

Status: not implemented yet.
"""
