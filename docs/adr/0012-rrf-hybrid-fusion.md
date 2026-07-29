# ADR 0012 — Reciprocal Rank Fusion for hybrid retrieval

**Status:** Accepted · **Date:** 2026-07-29

## Context

With two retrievers producing ranked lists, something has to combine them. The
obvious approach — blend the scores — does not survive contact with the data:

| Retriever | Score range | Meaning |
|---|---|---|
| BM25 | ~0 to 13, unbounded, corpus-dependent | Lexical overlap |
| Dense (cosine) | ~0 to 1, bounded | Angular similarity |

Adding them lets BM25 dominate by scale alone. Normalising first requires
knowing each retriever's distribution on *this* corpus, which changes whenever
the corpus does, plus a blend weight that must be re-tuned whenever either
retriever changes.

## Decision

**Reciprocal Rank Fusion** — discard the scores, keep only rank:

```
score(doc) = Σ over retrievers  1 / (k + rank_in_that_retriever)      k = 60
```

- **Rank is scale-free**, so the comparison problem disappears rather than
  being managed. A future retriever with no meaningful score at all still
  fuses.
- **k = 60** (the value from the original RRF paper). At k=60, rank 1
  contributes 1/61 ≈ 0.0164 and rank 2 contributes 1/62 ≈ 0.0161 — nearly
  identical. A large k flattens the curve, so appearing respectably in *both*
  retrievers beats topping one. That is deliberate: agreement between
  independent retrievers is stronger evidence than one retriever's confidence.
- **Identity is `(company, chunk_id)`.** `chunk_id` restarts per filing, so
  using it alone would merge Apple's chunk 42 with Tesla's — the same reasoning
  as the Qdrant point-ID scheme.
- **Ties break on the identity key**, not on insertion order. Discovered by a
  test: sorting on score alone left tied documents ordered by which retriever
  was passed first, so `fuse([dense, sparse])` and `fuse([sparse, dense])`
  could return different rankings. Fusion must be deterministic and symmetric.
- **Each retriever contributes `candidate_k=10`, wider than the final `top_k`.**
  Measured reason: in the dense baseline the answer to one query sat at rank 6.
  Ask each retriever for 5 and fusion never sees it.
- **The fused `score` overwrites whatever the source retriever wrote**, and
  `rrf_ranks` records the per-retriever ranks that produced it.

## Alternatives

**Weighted score blending** (`α·dense + (1-α)·sparse`, with normalisation).
Rejected above: fragile normalisation plus a hyperparameter to tune, and both
break when the corpus or either retriever changes. RRF has one constant that
did not need tuning.

**Cascade** — sparse to shortlist, dense to rank. Cheaper, and it inherits
sparse's recall ceiling: anything BM25 misses entirely is unrecoverable. The
point of hybrid is that either retriever can rescue the other.

**`EnsembleRetriever` from LangChain.** It does implement RRF, and this is a
place I deliberately wrote the ~40 lines myself: I wanted the identity key to
be `(company, chunk_id)` rather than page content, per-retriever ranks recorded
for debugging, and a deterministic tie-break. That last one is a real bug I
found in my own implementation via a test — worth knowing it exists whichever
implementation you use.

**Tuning k.** Not without more eval data. With 8 items, tuning k would be
fitting noise.

## Measured result — and it did not help

One index build, one harness, one dataset:

| Strategy | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| dense | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |
| sparse | 50.0% / 0.500 | 75.0% / 0.604 | 87.5% / 0.629 | 100% / 0.647 |
| **hybrid** | **75.0% / 0.750** | **87.5% / 0.812** | **87.5% / 0.812** | **100% / 0.833** |

**Hybrid exactly equals dense at every k.** That deserved a bug investigation
rather than a shrug, so:

1. `rrf_ranks` confirms both retrievers contribute — e.g. chunk 213 fused from
   dense rank 2 and sparse rank 1.
2. The fused *orderings* genuinely differ from dense. For
   `"derivative instruments"`: dense `[210, 213, 209, 211, 174, 127, …]`,
   hybrid `[213, 211, 210, 209, 193, 127, …]`.
3. Fusion is therefore working. The aggregate tie is a coincidence of an
   8-item dataset: hybrid reorders plenty, but the ranks *of the specific
   answer chunks* happen to land in the same places.

**The instructive part is why fusion could not fix the one failure.** For
`"derivative instruments"`, chunk 127 is at rank 6 in dense and rank 5 in
sparse. Both retrievers agree it is mediocre — and RRF rewards agreement. No
rank-combination scheme can promote a document that every input ranked
mid-pack.

That is a precise statement of what fusion cannot do, and it points at what is
needed instead: something that reads the query and the document *together*
rather than recombining opinions formed independently. A cross-encoder
reranker. Module 5 now has a specific, measured target rather than a hope.

## Consequences

- Hybrid is retained despite showing no gain here. It is a **variance
  reduction** on a corpus where lexical queries exist, and the eval set is far
  too small (8 items; one item is 12.5 points) to conclude it never helps.
- Hybrid costs both retrievers per query — roughly BM25's scoring pass on top
  of the embedding call. Latency is unchanged in practice because the embedding
  call dominates.
- `candidate_k` is now a real knob. Widening it costs almost nothing before the
  reranker exists and matters a great deal once it does.

## Interview angle

> **Q: How do you combine results from two retrievers?**
>
> Reciprocal Rank Fusion — I throw the scores away and keep only rank, summing
> `1/(60 + rank)` across retrievers. BM25 scores run 0 to 13 and cosine runs 0
> to 1, so adding them lets BM25 win on scale alone, and normalising means
> fitting a distribution that changes with the corpus. Rank is scale-free, so
> the problem disappears instead of being managed.
>
> `k=60` flattens the curve so agreement between retrievers beats one
> retriever's confidence — appearing second in both is worth more than first in
> one.
>
> **Follow-up: did it improve your numbers?**
>
> No. Hybrid matched dense exactly at every k, and I checked it was not a bug —
> the per-retriever ranks are recorded, both contribute, and the fused ordering
> genuinely differs. It is a coincidence of an eight-item dataset.
>
> What it did tell me is why fusion could not fix my one failure: the answer
> sits at rank 6 in dense and rank 5 in sparse, so both retrievers agree it is
> mediocre — and RRF rewards agreement. No way of recombining those two
> rankings promotes it. That needs a cross-encoder that reads the query and the
> document together, which is exactly what I built next. The negative result
> was more useful than a positive one would have been.
>
> **Follow-up: why not `EnsembleRetriever`?**
>
> I wanted the identity key to be `(company, chunk_id)` rather than content,
> the per-retriever ranks kept for debugging, and deterministic tie-breaking.
> That last one was a real bug in my first version — ties resolved by whichever
> retriever I passed first, so fusing the same two lists in the other order
> gave a different ranking. A test caught it.
