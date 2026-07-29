# Interview — Module 4: Sparse retrieval and hybrid fusion

The module with a **negative result**, which makes it the most useful one to
talk about. Anyone can present a graph that goes up.

---

## Q1. Why keep BM25 when you have embeddings?

**Answer.** Because they fail in opposite directions.

Embeddings *generalise* — which is why dense retrieval finds "net sales" when
you ask about "revenue", and also why it returned eleven chunks about
derivative instruments in essentially arbitrary order with the answer at rank
6. BM25 does not generalise at all, but a rare exact token like `Megapack` or
`Item 1A` is decisive for it.

On my eval set BM25 alone is **worse than dense at every k** — 50% versus 75%
at k=1, 87.5% versus 87.5% only by k=5. That is the expected result, because
the dataset is mostly natural-language questions whose wording differs from the
filing's. I would not ship it alone. Its value is that its errors are
uncorrelated with dense's.

**Follow-up: what does BM25 actually compute?**

Per query term: higher when the term appears often in a document (term
frequency, with saturation so the tenth occurrence adds little), higher when
the term is rare across the corpus (inverse document frequency), adjusted for
document length so long chunks are not rewarded for size. No training, no
vectors, no API call.

---

## Q2. What is the easiest way to get BM25 wrong?

**Answer.** Tokenizer asymmetry. If the corpus is tokenised one way at index
time and the query another, matching degrades silently — no error, just worse
results. Lowercase one side only and `Megapack` never matches `megapack`.

There is exactly one `tokenize()` in the codebase, both paths call it, and a
test asserts `tokenize("Megapack") == tokenize("megapack")`.

**Follow-up: no stemming or stop words?**

Deliberately. Stemming would let `hedges` match `hedging` but would also
collapse distinct financial terms. And BM25's IDF already discounts words that
appear everywhere, which is what stop-word lists usually exist for — the
mechanism is built in.

There is a real cost I accept: `$416,161` tokenises to `416` and `161`, so
BM25 cannot match a dollar figure as a unit. The row label beside it carries
the meaning.

---

## Q3. How do you combine two retrievers?

**Answer.** Reciprocal Rank Fusion — throw the scores away, keep only rank:

```
score(doc) = Σ  1 / (60 + rank_in_that_retriever)
```

The reason is a scale mismatch that does not resolve cleanly. BM25 scores run
roughly 0 to 13 and are corpus-dependent; cosine similarity runs 0 to 1. Adding
them lets BM25 win on scale alone. Normalising first means fitting each
retriever's distribution on this corpus — which changes whenever the corpus
does — plus a blend weight to re-tune whenever either retriever changes.

Rank is scale-free, so the problem disappears rather than being managed.

**Follow-up: why k=60?**

It flattens the curve. At k=60 rank 1 contributes 1/61 and rank 2 contributes
1/62 — nearly identical — so appearing respectably in *both* retrievers beats
topping one. That encodes a judgement: agreement between independent retrievers
is stronger evidence than one retriever's confidence. At k=1 the top rank
dominates and a single confident retriever wins. There is a test demonstrating
exactly that flip.

**Follow-up: why not `EnsembleRetriever`?**

It does implement RRF. I wrote the ~40 lines because I wanted the identity key
to be `(company, chunk_id)` rather than page content, per-retriever ranks
recorded for debugging, and deterministic tie-breaking — see Q5.

---

## Q4. Did hybrid improve your numbers?

**No. Say so plainly — this is the answer that separates you from a
demo.**

| Strategy | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| dense | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |
| sparse | 50.0% / 0.500 | 75.0% / 0.604 | 87.5% / 0.629 | 100% / 0.647 |
| hybrid | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |

Hybrid matched dense exactly at every k. An exact tie is suspicious, so I
checked it was not a bug:

1. `rrf_ranks` metadata confirms both retrievers contribute — chunk 213 fused
   from dense rank 2 and sparse rank 1.
2. The fused ordering genuinely differs: dense `[210, 213, 209, 211, …]`,
   hybrid `[213, 211, 210, 209, …]`.
3. So fusion works. The aggregate tie is a coincidence of an 8-item dataset —
   hybrid reorders plenty, but the ranks of the specific *answer* chunks land
   in the same places.

**Follow-up: so was it a waste?**

No, because of *why* it could not help. For the failing query, the answer is at
rank 6 in dense and rank 5 in sparse. **Both retrievers agree it is mediocre —
and RRF rewards agreement.** No scheme that recombines those two rankings can
promote a document that every input ranked mid-pack.

That is a precise statement of what fusion cannot do, and it points at what is
needed: something that reads the query and the document *together* rather than
recombining opinions formed independently. A cross-encoder reranker. The
negative result gave Module 5 a specific measured target instead of a hope.

I kept hybrid because it is variance reduction on a corpus where lexical
queries exist, and eight items — where one item is 12.5 points — cannot support
the claim that it never helps.

---

## Q5. Tell me about a bug your tests caught.

**Answer.** RRF ties were resolving by dict insertion order, which depends on
the order retrievers were passed in. So `fuse([dense, sparse])` and
`fuse([sparse, dense])` could return different rankings for identical inputs.

Ties are not rare — two documents at the same rank in different retrievers
score identically by construction.

I only found it because I wrote a test asserting fusion is symmetric in
retriever order, which I wrote because RRF *should* be symmetric mathematically.
The fix is breaking ties on the identity key so the result is deterministic and
genuinely order-independent.

It matters more than it sounds: non-determinism makes two eval runs
incomparable, and the harness exists precisely to compare runs.

---

## Q6. Why is the candidate pool wider than the final top_k?

**Answer.** Each retriever contributes 10 candidates before fusion slices to
the final k. The reason is measured, not assumed: in the dense baseline the
answer to one query sat at **rank 6**. Ask each retriever for 5 and fusion
never sees it.

The general rule: **never slice to the final `top_k` before fusion or
reranking.** A chunk cut early cannot be recovered by any downstream stage,
however good. That becomes critical in the next module, where the reranker only
ever sees what fusion passed it.

---

## Q7. Where does the in-memory BM25 index break?

**Answer.** It is the first thing that breaks as the corpus grows. It is
rebuilt from the whole corpus at process start — seconds at 1,309 chunks, a
multi-minute startup at a million — and it makes the service stateful, so every
replica pays the cost and horizontal scaling gets worse.

The fix is Qdrant's native sparse vectors: the sparse index lives beside the
dense one in the same collection, so filtering, persistence and scaling all
come free.

**Follow-up: why not do that now?**

Because it would conflate two changes — adopting sparse retrieval, and moving
it into the store. I want a baseline for BM25 as its own thing first, so that
when I move it I can show nothing regressed. Same reasoning as not tuning chunk
size before I had a retrieval number.

---

## Q8. What is weak here?

1. **Eight eval items.** Every conclusion in this module is provisional. An
   exact tie between hybrid and dense is exactly what a too-small dataset
   produces.
2. **In-memory BM25**, as above.
3. **`k=60` is untuned.** With 8 items, tuning it would be fitting noise.
4. **Corpus-wide IDF** rather than per-company. Arguably wrong for
   company-scoped queries; no evidence yet that it matters.
5. **No query preprocessing** — no expansion, no multi-query. That is another
   lever on the same vocabulary-mismatch problem, and untested.
