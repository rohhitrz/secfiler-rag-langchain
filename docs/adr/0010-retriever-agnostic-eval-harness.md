# ADR 0010 — A retriever-agnostic evaluation harness

**Status:** Accepted · **Date:** 2026-07-29

## Context

Every later module in this project claims an improvement: hybrid search beats
dense, reranking beats fusion, contextual retrieval beats plain chunks. Those
claims are worthless unless the same measurement applies to all of them.

The temptation is to let the harness help. It has the dataset, it knows each
item names a company, so it could apply the company filter itself — and then it
needs to know how each retriever expects filters, and then a strategy that
filters differently needs a special case, and at that point the harness is part
of the system under test.

The previous build got this right and recorded it as frozen. This ADR restates
the constraint and adds what was missing: auditability.

## Decision

**1. The harness accepts a `SearchFn`, not a retriever object.**

```python
SearchFn = Callable[[str, Mapping[str, Any], int], Sequence[Document]]
#                    query  filters            top_k
```

Anything satisfying that signature can be scored. The harness does not know
whether it is calling BM25, Qdrant, or a five-stage pipeline.

**2. Filters are opaque.** They travel from dataset to retriever as a mapping
the harness forwards and never inspects. The *dataset loader* knows the schema;
the *retriever* knows what `company` means; the harness knows neither. A test
asserts this directly — `test_harness_never_inspects_filters`.

**3. Ground truth is a substring, not a chunk ID.** Chunk IDs renumber whenever
the chunker changes, which would silently invalidate the dataset on any
ingestion tweak. A substring of cleaned text survives re-chunking, so a
1000-character chunker and an 800-character one can be compared with the same
data.

**4. Two metrics, always reported together.**

- **Hit rate @ k** — did the right chunk reach the context at all? A ceiling on
  answer quality.
- **MRR** — *where* did it land? Models attend most reliably to the start and
  end of their context.

A reranker typically leaves hit rate flat and moves MRR. Reporting hit rate
alone would make it look like the reranker did nothing.

**5. Two tiers, reported separately.** Tier 1 is a near-tautological smoke test;
tier 2 is realistic phrasing. A blended number lets a tier-2 regression hide
behind tier-1 passes.

**6. Every result carries the chunk that matched.** `matched_rank`,
`matched_chunk_id` and `matched_excerpt()` exist so a pass can be *read*. This
is the lesson that cost the previous build two false positives: a loose
substring like `"net sales"` matches dozens of chunks and reports a hit while
retrieval actually missed.

## Alternatives

**Take a LangChain `BaseRetriever` instead of a callable.** More idiomatic, and
rejected for a concrete reason: `BaseRetriever.invoke()` takes only a query, so
per-item filters must be baked in at construction — one retriever object per
company. The eval dataset cannot express that, and a hybrid strategy that
filters at two different stages cannot either. The callable is the smaller
interface, and `DenseRetriever.as_langchain_retriever()` still exists for LCEL
composition where filters are fixed.

**LangSmith evaluators.** Genuinely useful, and scheduled for the observability
module for *generation* metrics like faithfulness, where an LLM judge is the
only practical option. Rejected as the primary retrieval harness: retrieval
correctness here is a deterministic substring check, and making it depend on a
network service and an account would mean the core quality gate cannot run
offline or in CI.

**Exact chunk-ID ground truth.** More precise, and it would eliminate the false
positives entirely. Rejected because it makes the dataset a hostage to the
chunker: change the chunk size and every ID is wrong, so the dataset must be
rebuilt exactly when you most want to compare before and after. The audit
output is the mitigation for the looseness this accepts.

**Recall@k and nDCG.** nDCG needs graded relevance judgements, which this
dataset does not have (an item is a hit or it is not). Recall@k equals hit rate
when there is one relevant chunk per query. Both become worth adding when the
dataset grows to multiple relevant chunks per question.

## Consequences

- Any future retriever is scored by writing a five-line adapter, and its number
  is directly comparable to every earlier number.
- Substring ground truth admits false positives. Mitigated by the audit output,
  and by the discipline of reading it. **A green number nobody has read is not
  evidence.**
- Eight items is a thin dataset. Every number in this project should be read
  with that in mind — a single item is 12.5 percentage points.
- Editing an item invalidates every score recorded before the edit. The dataset
  is versioned for this reason, and re-baselining must be stated explicitly.

## Baseline established (dense retrieval, 1,309 chunks, `text-embedding-3-small`)

| k | Hit rate | MRR |
|---|---|---|
| 1 | 75.0% | 0.750 |
| 3 | 87.5% | 0.812 |
| 5 | 87.5% | 0.812 |
| 10 | **100.0%** | 0.833 |

**Read that curve carefully — it is the design brief for the next two modules.**
Recall is *solved* at k=10: every answer is in the candidate pool. MRR barely
moves (0.812 → 0.833), which means those chunks are present but badly ordered.

So the gap is not recall, it is **precision at the top**. That is a reranker's
job, not a better embedding model's — and now it is a measured claim rather
than an assumption.

## Interview angle

> **Q: How do you know your RAG system is any good?**
>
> A harness that takes any retrieval function and a fixed dataset and reports
> hit rate and MRR. The design constraint that matters is that it knows nothing
> about the strategy it is scoring — filters pass through it as an opaque
> mapping it never inspects. The moment the harness special-cases one
> retriever, its numbers stop being comparable across the others, and every A/B
> after that measures the harness.
>
> The current dense baseline is 87.5% hit rate at k=5, 100% at k=10, with MRR
> at 0.833. That gap is the interesting part: recall is solved, ranking is not.
> The right chunk is in the pool but sitting at rank 6 — which tells me to
> invest in reranking, not in a better embedding model.
>
> **Follow-up: substring matching seems fragile.**
>
> It is, deliberately. Chunk-ID ground truth is more precise but renumbers
> whenever the chunker changes, so the dataset breaks exactly when you want to
> compare before and after. I took the looseness and paid for it with an audit
> mode that prints the matching chunk for every pass — my previous build had
> two false positives from a loose substring, and reading the matches is what
> caught them.
>
> **Follow-up: eight items is not much.**
>
> No, and I say so wherever a number appears — one item is 12.5 points. It is
> enough to catch a broken pipeline and to rank strategies coarsely; it is not
> enough to justify a small improvement. Growing it to 25+ hand-written pairs
> is scheduled before any claim that depends on single-digit differences.
