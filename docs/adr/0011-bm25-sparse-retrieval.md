# ADR 0011 — BM25 sparse retrieval with a shared tokenizer

**Status:** Accepted · **Date:** 2026-07-29

## Context

Dense retrieval hit 87.5% at k=5 with one failure: the bare keyword query
`"derivative instruments"`, where eleven Apple chunks are near-identical in
embedding space and the answer sat at rank 6.

That is the shape of failure embeddings own. They generalise — which is what
makes them find "net sales" when you ask about "revenue", and also what makes
them blur eleven chunks about the same topic into an arbitrary order. Rare,
exact tokens (`Megapack`, `Powerwall`, `Item 1A`, a specific figure) are
precisely what a lexical retriever handles decisively and an embedding
smooths away.

## Decision

Add BM25 (`rank-bm25`, `BM25Okapi`) as a second retrieval strategy with the
same `(query, filters, top_k) -> list[Document]` shape as dense.

**1. Exactly one tokenizer, used for both indexing and querying.**

```python
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())
```

Symmetry is not a nicety. Lowercase one side only and `Megapack` never matches
`megapack`; split punctuation differently and the retriever silently finds
less. There is one function and both paths call it, with a test asserting the
property directly.

**2. No stemming, no stop words.** Stemming would let `hedges` match `hedging`
but would also collapse distinct financial terms. BM25's IDF already discounts
words appearing everywhere, which is what stop-word lists usually exist for.

**3. One corpus-wide index; filters applied after scoring.** IDF is therefore
computed across all three filings rather than per company.

**4. Documents scoring zero are excluded.** A chunk sharing no query term is
not a weak match, it is no match — returning it would pad the candidate list
with noise that fusion then has to rank.

**5. The index lives in process memory, built once at construction.**

## Alternatives

**Per-company BM25 indexes.** Arguably more correct for company-scoped queries:
IDF would reflect one filing, so a term common in Apple's 10-K but rare in
Tesla's would score appropriately in each. Rejected for now because scores
would no longer be comparable across companies, unfiltered queries would need a
separate corpus-wide index anyway, and the eval set gives no evidence the
difference matters here. Worth revisiting with a measurement.

**Qdrant native sparse vectors.** The strategically better answer, and probably
where this ends up. Sparse vectors live beside the dense ones in the same
collection, so filtering, scaling and persistence all come for free, and the
in-memory index — which must be rebuilt at every process start — disappears.
Rejected *for this module* because it would conflate two changes: adopting
sparse retrieval, and moving it into the store. Baseline first, then move it
with a number proving nothing regressed.

**Stemming or a full lexical engine (OpenSearch).** Both are real answers at
scale. Neither is justified by 1,309 chunks.

## Consequences

- The BM25 index is rebuilt at every process start — seconds at 1,309 chunks,
  a multi-minute startup at a million. **This is the first thing that breaks as
  the corpus grows**, and the fix is Qdrant sparse vectors, not a bigger box.
- Tokenising `$416,161` yields `416` and `161`. Acceptable because the
  surrounding row label carries the meaning, but it means BM25 cannot match a
  dollar figure as a unit.
- Sparse retrieval needs the chunk *texts*, which the vector store does not
  conveniently provide — so any caller wanting hybrid search must hold the
  corpus in memory too. Visible in the eval CLI, which ingests for BM25 even
  when querying a live Qdrant.

## Measured result

| Strategy | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| dense | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |
| **sparse** | **50.0% / 0.500** | **75.0% / 0.604** | **87.5% / 0.629** | **100% / 0.647** |

BM25 alone is **worse than dense at every k** — as expected, and worth stating
plainly. The eval set is dominated by natural-language questions whose wording
differs from the filing's, which is exactly dense retrieval's strength.

Its value is not as a replacement. It is as a retriever that fails
*differently*, which only pays off through fusion — see
[ADR 0012](0012-rrf-hybrid-fusion.md), including the honest finding that on
this dataset it did not pay off yet.

## Interview angle

> **Q: Why keep BM25 when you have embeddings?**
>
> Because they fail in opposite directions. Embeddings generalise, which is why
> they find "net sales" when you ask about "revenue" — and also why they
> returned eleven chunks about derivative instruments in essentially arbitrary
> order and put the answer at rank 6. BM25 does not generalise at all, but a
> rare exact token like `Megapack` is decisive for it.
>
> On my eval set BM25 alone is *worse* than dense at every k — 50% versus 75%
> at k=1. That is the expected result and I would not ship it alone. The reason
> to have it is that its errors are uncorrelated with dense's.
>
> **Follow-up: what is the easiest way to get BM25 wrong?**
>
> Tokenizer asymmetry. If the corpus is tokenised one way at index time and the
> query another, matching silently degrades — no error, just worse results.
> There is exactly one `tokenize()` in my codebase and a test asserting
> `tokenize("Megapack") == tokenize("megapack")`.
>
> **Follow-up: in-memory BM25 at scale?**
>
> It breaks first. The index is rebuilt from the whole corpus at process start —
> fine at 1,300 chunks, a multi-minute startup at a million, and it makes the
> service stateful so replicas each pay the cost. The fix is Qdrant's native
> sparse vectors, which put the sparse index beside the dense one. I kept it
> in-memory deliberately for this module so that adopting BM25 and moving it
> into the store stay two separately measurable changes.
