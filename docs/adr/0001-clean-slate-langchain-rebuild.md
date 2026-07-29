# ADR 0001 — Rebuild from a clean slate on LangChain

**Status:** Accepted · **Date:** 2026-07-29

## Context

A working raw-Python RAG system already existed: BM25 baseline, Qdrant vector
search, RRF hybrid fusion, a Cohere reranker, and an eval harness scoring 8/8.
It worked, and it taught the underlying mechanics — which is exactly why it was
built that way first.

But it had accumulated the shape of exploratory work:

- `fuse.py` held a pure fusion function, the hybrid wiring, the reranker wiring
  *and* a `__main__` block — four responsibilities in one file
- API clients (`OpenAI()`, `QdrantClient()`, `cohere.ClientV2()`) were
  constructed at module import, so importing anything required live credentials
- Pipelines were driven by `if __name__ == "__main__"` blocks rather than a
  callable API
- Zero tests
- Generated artifacts (`*_clean.txt`, `scratch_bm25.py`, `qdrant_storage/`)
  lived beside source

The goal for this repo is different from the goal of that one: not "learn how
retrieval works" but "demonstrate production engineering around retrieval."

## Decision

Start this repository from a clean slate and rebuild on **LangChain**, keeping
the *decisions* from the previous build and discarding the *code*.

Carried forward (measured or reasoned, not re-litigated):

- Lowercase company keys everywhere
- Single Qdrant collection + payload filter, not per-company collections
- Deterministic `uuid5` point IDs
- RRF (k=60) with `(company, chunk_id)` as the identity key
- Retriever-agnostic eval harness
- The 10 → 10 → 3 candidate funnel

Discarded: every line of the previous implementation. The old code remains in
git history and in the sibling `secfiler-rag` repository.

## Alternatives

**Incrementally refactor the existing code into LangChain.** Rejected: the
modules would have to be rewritten to fit LangChain's `Document` /
`BaseRetriever` interfaces anyway, and incremental refactors of exploratory
code tend to preserve its structure. Starting clean costs less than it looks.

**Keep raw Python, skip LangChain.** Rejected for this repo's purpose.
LangChain provides the interfaces that make components swappable
(`BaseRetriever`, `Document`, LCEL composition) and LangSmith tracing for free
— and it is what the target roles use. The previous build already proved the
mechanics are understood without it.

**Use LlamaIndex instead.** Reasonable alternative, strong at ingestion and
indexing abstractions. Rejected because LangChain's ecosystem and LangSmith
integration are more widely used in production job descriptions, and because a
lower-level framework leaves more of the RAG design visibly ours.

## Consequences

- Nothing ships on day one; the foundation is built before features.
- A framework dependency is accepted, along with its churn.
- **Mitigation for the main risk** — that the framework hides the concepts —
  is explicit: every module's docs state what LangChain abstracts and what the
  underlying concept is. Chunking strategy, fusion, filtering and eval design
  stay ours; LangChain does orchestration and I/O.

## Interview angle

> **Q: Why rewrite something that worked?**
>
> It worked as a learning artifact and it proved the retrieval decisions, which
> I carried over. What it did not have was structure I could defend: one file
> with four responsibilities, import-time API clients that made testing
> impossible, and no tests at all. The rewrite is about the engineering around
> the RAG, not the RAG itself — and the fact that I could carry the retrieval
> decisions forward unchanged is evidence they were the right ones.
>
> **Follow-up: doesn't LangChain hide the concepts you learned?**
>
> It would if I had started there. I built BM25, RRF fusion and the reranker
> integration by hand first, so I know what `EnsembleRetriever` does internally
> and where its defaults would hurt me. That order — mechanics first, framework
> second — is deliberate.
