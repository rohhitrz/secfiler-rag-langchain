# ADR 0013 — Cross-encoder reranking, and fail-open as a measurement hazard

**Status:** Accepted · **Date:** 2026-07-29

## Context

[ADR 0012](0012-rrf-hybrid-fusion.md) ended with a precise, measured problem.
For the query `"derivative instruments"`, the answer (Apple chunk 127) sat at
**rank 6 in dense and rank 5 in sparse**. Both retrievers agreed it was
mediocre, and RRF rewards agreement — so no way of recombining those two
rankings could promote it. Hybrid tied dense at every k.

The limit is structural, not a tuning problem. Dense retrieval compares two
vectors computed **independently**: the query never saw the document, and the
document never saw the query. A bi-encoder must compress a whole chunk into one
vector before knowing what will be asked of it. BM25 is lexical overlap and
knows even less.

A **cross-encoder** reads the query and document *together* in one forward pass
and scores that pair directly. It cannot be precomputed — so it is far too
expensive over a corpus, and markedly more accurate over a shortlist.

## Decision

**1. Add `RerankingRetriever`, wrapping any retriever.** It takes
`candidate_k` results from its base, reranks, and returns `top_k`. Because it
composes with dense, sparse or hybrid, the harness can measure
rerank-over-dense against rerank-over-hybrid and answer whether fusion still
earns its place.

**2. Cohere `rerank-v3.5` behind a `RerankClient` Protocol.** The concrete
client is never imported by the retriever, so tests inject a fake and a
provider swap touches one function.

**3. Index-alignment discipline.** Map `result.index` — a position into the
list we *sent* — back into the original candidate list. Never pair by position
in the response. Cohere does return results sorted by relevance, but relying on
that would silently mispair documents and scores if it ever changed, and the
output would still look like a plausible ranking. Enforced by a test whose fake
deliberately returns results in reverse-score order.

**4. The funnel is `retrievers → 10 → rerank → 3`.** Reranking reorders; it
cannot retrieve. A chunk outside `candidate_k` is unrecoverable.

**5. `fail_open=True` in a service, `fail_open=False` when measuring.**

**6. Bounded retry with exponential backoff** on rate limits and timeouts.

## The finding that mattered most

The first measurement run reported hybrid+rerank as *identical* to hybrid.
Investigating produced a genuinely uncomfortable result: **the reranker had
never run.**

Cohere's trial tier allows roughly ten requests per minute. An eval run issues
dozens back to back, so nearly every call returned `TooManyRequestsError` —
and `fail_open=True` did exactly what it was designed to do: logged a warning
and returned the un-reranked candidates. The harness dutifully scored the
fallback and reported it as the reranker's number.

**In a service that behaviour is correct.** Reranking improves ordering; the
system is still right without it, and failing a user's query over an optional
enhancement is worse than serving a slightly worse ranking.

**In an evaluation it is a trap**, and a subtle one: the run completes, the
numbers look plausible, and the only visible signal is a warning line that
carries its detail in a structured `extra` field the console formatter does not
print. You cannot measure a component that is quietly not running.

Two changes followed:

- The eval CLI constructs the reranker with `fail_open=False`. Measurement
  fails loudly.
- Provider exceptions are wrapped in `RetrievalError`, so a caller catching
  `SecfilerRagError` genuinely catches everything this package raises. A raw
  `cohere.TooManyRequestsError` escaping `search()` broke that contract.

## Alternatives

**Self-hosted `bge-reranker-large`.** No rate limit, no per-query cost, no
third party in the request path — and it needs a GPU to be fast, plus model
serving to operate. The right answer at volume; the wrong one for a project
where the reranker must be reproducible from a clone.

**LangChain's `CohereRerank` / `ContextualCompressionRetriever`.** The
idiomatic wrapper. Not used because it is a `BaseRetriever`, which takes only a
query — the same per-query-filter problem as
[ADR 0010](0010-retriever-agnostic-eval-harness.md) — and because it hides both
the index-alignment mapping and the failure policy, which are the two things
here worth being explicit about.

**Rerank the whole corpus.** ~1,309 cross-encoder calls per query. Not an
option, and the reason the funnel exists.

**No reranker; widen `top_k` instead.** Recall was already 100% at k=10, so
widening "works" — by sending seven irrelevant chunks to the LLM. That costs
tokens and triggers lost-in-the-middle. The point is *precision at the top*.

## Measured result

Same index, same harness, same dataset, `top_k=3`:

| Strategy | Hit rate | MRR | Median latency |
|---|---|---|---|
| dense | 87.5% | 0.812 | 464 ms |
| hybrid | 87.5% | 0.812 | 497 ms |
| **dense+rerank** | **100.0%** | **0.917** | 886 ms |
| **hybrid+rerank** | **100.0%** | **0.917** | — |

**The prediction held.** Audited rather than assumed: chunk 127 moved from
rank 6 to **rank 3**, and its excerpt is the exact expected phrase — *"The
Company uses derivative instruments, such as foreign currency forward and
option contrac…"*. Tier 1 went 80% → 100%.

A second improvement showed up in the audit that the aggregate number hides:
*"what was Apple's total revenue this year?"* previously matched chunk 201; it
now matches **chunk 178**, the actual consolidated statement of operations —
`Products | $307,003 | $294,866 | $298,085`. A materially better chunk, and one
that only contains figures at all because of the inline-XBRL fix in
[ADR 0007](0007-preserve-inline-xbrl-values.md).

**And `dense+rerank` exactly equals `hybrid+rerank`.** On this dataset, once a
reranker exists, **fusion adds nothing** — hybrid costs a second retriever per
query for zero measured gain. Sparse retrieval is retained because eight items
cannot support "never", but the honest current recommendation is
**dense + rerank**.

## Consequences

- Retrieval latency roughly doubles (464 ms → 886 ms). Irrelevant once
  generation lands at 1–4 seconds, and it buys the precision that lets the
  prompt carry 3 chunks instead of 10 — which is cheaper *and* more accurate.
- A per-query paid dependency in the read path, with a rate limit that is a
  real operational constraint, not a footnote.
- The reranker is the only optional stage. That is now an explicit, tested
  property (`fail_open`), not an accident.
- Hybrid's value is unproven once reranking exists. Flagged rather than
  removed, pending a larger eval set.

## Interview angle

> **Q: Why add a reranker when your recall was already 100% at k=10?**
>
> Because recall was not the problem — ranking was. The right chunk was in the
> pool at rank 6, and I could have "fixed" it by sending ten chunks to the LLM
> instead of three. That costs tokens and triggers lost-in-the-middle, where
> models under-weight material in the centre of their context. A cross-encoder
> reads the query and document together instead of comparing two independently
> computed vectors, so it can separate eleven chunks that are near-identical in
> embedding space. It took hit rate from 87.5% to 100% and MRR from 0.812 to
> 0.917, and I verified the specific chunk moved from rank 6 to rank 3 rather
> than trusting the aggregate.
>
> **Q: Tell me about a bug that taught you something.**
>
> My first rerank measurement showed no improvement at all. The reranker had
> never run: Cohere's trial tier allows about ten calls a minute, my eval
> issues dozens, and I had built the reranker to fail open — log a warning and
> return the un-reranked results. That is the *correct* production behaviour,
> because reranking is an enhancement and failing a user's query over it would
> be worse. But in an evaluation it meant I measured the fallback and would
> have reported it as the reranker's score.
>
> The fix was two things: retry with exponential backoff, which any per-request
> paid API needs anyway, and `fail_open=False` in the eval path. The principle
> I took from it is that **graceful degradation and honest measurement are in
> direct conflict**, and the same component needs opposite policies in the two
> contexts.
>
> **Follow-up: is hybrid search still worth it?**
>
> On my current numbers, no — `dense+rerank` and `hybrid+rerank` are identical,
> so BM25 is costing a second retriever per query for nothing. I have not
> removed it, because eight eval items cannot support a claim that strong, but
> if I had to ship today I would ship dense plus reranking and say so.
