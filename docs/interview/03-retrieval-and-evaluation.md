# Interview — Module 3: Dense retrieval and evaluation

The first module that produces a **number**. Everything before this was
plumbing; this is where claims become measurable.

---

## Q1. How do you know your RAG system is any good?

**The question this whole module exists to answer.**

A harness that takes any retrieval function plus a fixed dataset and reports
hit rate and MRR. The design constraint that matters: **it knows nothing about
the strategy it is scoring.** Filters travel through it as an opaque mapping it
never inspects.

The moment the harness special-cases one retriever — "if this is BM25,
lowercase the query" — its numbers stop being comparable across the others, and
every A/B after that is measuring the harness.

Current dense baseline over 1,309 chunks:

| k | Hit rate | MRR |
|---|---|---|
| 1 | 75.0% | 0.750 |
| 3 | 87.5% | 0.812 |
| 5 | 87.5% | 0.812 |
| 10 | 100.0% | 0.833 |

**Follow-up: what does that curve tell you?**

That recall is solved and ranking is not. By k=10 every answer is in the
candidate pool, but MRR has barely moved — 0.812 to 0.833. The right chunks are
present and badly ordered.

So the next investment is **reranking, not a better embedding model.** That is
a measured conclusion, and it is the opposite of what I would have guessed:
"retrieval is missing things" is the intuitive diagnosis, and it is wrong here.

---

## Q2. Why hit rate *and* MRR?

**Answer.** They answer different questions and each hides something the other
sees.

Hit rate asks: did the right chunk reach the context at all? It is the ceiling
on answer quality — a chunk that never reaches the prompt cannot be used.

MRR asks: *where* did it land? Position matters because models attend most
reliably to the beginning and end of their context, so rank 5 is worth less
than rank 1.

The concrete case: **a reranker typically leaves hit rate flat and moves MRR.**
It does not find new chunks, it reorders the ones you have. With hit rate alone
the reranker would look like it did nothing.

---

## Q3. Why substring ground truth instead of chunk IDs?

**Answer.** Chunk IDs renumber whenever the chunker changes. So the moment you
tune chunk size — exactly when you most want a before-and-after number — every
ID in the dataset is wrong and the dataset must be rebuilt. You cannot measure
the change you just made.

A substring of the cleaned text survives re-chunking, so a 1000-character
chunker and an 800-character one can be compared with the same data.

**Follow-up: doesn't that produce false positives?**

Yes, and that is the accepted cost. A loose substring like `"net sales"` can
match dozens of chunks and report a hit while retrieval actually missed. My
previous build had two such false positives.

The mitigation is structural: every result records the rank, chunk ID and an
excerpt of what matched, and the CLI has an `--audit` mode that prints them.
**A green number nobody has read is not evidence.** That is a habit, not a
feature — but the feature makes the habit cheap.

---

## Q4. Walk me through a miss you diagnosed.

**The best answer in this module, because it is real.**

The seed set has a tier-1 item: query `"derivative instruments"`, expecting
`"uses derivative instruments"`. It missed at k=5.

First question: is the eval item even valid? I checked the cleaned corpus —
the phrase appears exactly once. So the item is fair and retrieval genuinely
missed. **That check matters: if the substring had not existed, I would have
been about to "fix" retrieval for an eval bug.**

Then: 11 Apple chunks mention "derivative instruments". The target sits at
**rank 6** — just outside the window. Widening to k=20 or k=50 does not move
it; rank 6 is where dense similarity puts it.

The diagnosis is retrieval noise, not a retrieval failure: a bare keyword query
where eleven chunks are near-identical in embedding space, so the ranking among
them is close to arbitrary.

Two things confirmed it. Rephrasing as a real question — *"Does Apple use
derivative instruments to hedge foreign currency risk?"* — pulls the same chunk
to **rank 2**. And the tier-2 items, all natural-language questions, score 100%
at k=3 while this tier-1 keyword query fails.

**Follow-up: so what do you do about it?**

Nothing yet, deliberately. It tells me what to build — a reranker that reads
query and document *together* can separate eleven near-identical chunks in a
way that comparing two independently-computed vectors cannot. I now have a
baseline to measure that against, which I would not have had if I had started
tuning.

---

## Q5. Why isn't your retriever a LangChain `BaseRetriever`?

**Answer.** `BaseRetriever.invoke()` takes only a query string, so per-query
filters have to be baked in at construction — one retriever object per company.
The eval dataset cannot express that: each item carries its own company, and
the harness is not allowed to know what a company is.

So `DenseRetriever` exposes `search(query, filters, top_k)`, and
`as_langchain_retriever()` provides the framework-native adapter for LCEL
chains where the filter is fixed at build time.

That is the general principle again: **use the framework interface where it
fits, and do not contort the design to satisfy it where it does not.**

---

## Q6. What actually happens when you call `similarity_search`?

**Answer.** The query is embedded with the *same* model used at index time,
then Qdrant finds the nearest stored vectors by cosine similarity using its
HNSW graph — an **approximate** nearest-neighbour search, not an exhaustive
scan. That is the trade: sub-linear latency for a small chance of missing a
true nearest neighbour.

The part worth stressing is **symmetry**. Query and documents must be embedded
by the same model with the same dimensions. Mixing models produces vectors in
unrelated spaces, and the failure is not an error — it is retrieval returning
confident nonsense.

**Follow-up: what would you tune in HNSW?**

`m` and `ef_construct` at build time, `ef` at search time — all trading recall
against latency and memory. Not yet, though: at 1,309 vectors the index is
effectively exhaustive anyway, so tuning it would be measuring noise.

---

## Q7. Why does an unknown filter key raise?

**Answer.** Because the alternative is silent. If `{"sector": "tech"}` were
ignored, the query would return chunks from every company and look like a
retrieval *quality* problem — you would go tune embeddings to fix a typo.

Same reasoning as the lowercase company convention in ingestion: the failure
modes that cost real time are the ones that return plausible wrong answers
instead of errors.

---

## Q8. What is the median latency and does it matter?

**Answer.** ~440 ms per query, which is almost entirely the OpenAI embedding
call — the Qdrant search itself is single-digit milliseconds at this scale.

It does not matter yet. Once generation lands, the LLM call will be 1–4 seconds
and dominate everything. Optimising a 440 ms embedding call under a 2-second
completion is invisible to users. Caching query embeddings is the obvious win
when it does matter, and it is cheap because it is deterministic.

---

## Q9. What is weak about this evaluation?

1. **Eight items.** One item is 12.5 percentage points. Enough to catch a
   broken pipeline and rank strategies coarsely; not enough to justify a small
   improvement. Growing to 25+ is scheduled before any such claim.
2. **One relevant chunk per query.** Recall@k and nDCG only become meaningful
   with graded or multiple relevance judgements.
3. **Retrieval only.** Faithfulness and answer relevance need generation and an
   LLM judge.
4. **No cross-company queries.** "Compare Apple and Tesla" needs multi-filter
   retrieval and a different success metric entirely.
5. **`"net sales"` is a loose substring** and could pass by accident. It is
   flagged in the dataset file rather than quietly tolerated.
