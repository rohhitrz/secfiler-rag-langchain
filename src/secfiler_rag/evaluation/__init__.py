"""Cross-cutting — measurement infrastructure.

Responsibility: the retriever-agnostic eval harness and its metrics
(hit-rate, MRR, later faithfulness / answer-relevance via LangSmith).

Hard rule carried over from the previous build: the harness never learns domain
knowledge. It receives a retriever and a dataset; it must not know that BM25,
Qdrant, or reranking exist. That constraint is what makes A/B numbers honest.

Status: not implemented yet.
"""
