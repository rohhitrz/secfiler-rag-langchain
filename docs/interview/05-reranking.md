# Interview — Module 5: Cross-encoder reranking

The module where a measured prediction came true, and where a bug taught a
principle worth more than the feature.

---

## Q1. Why add a reranker when recall was already 100% at k=10?

**Answer.** Because recall was not the problem — ranking was.

The right chunk was already in the pool at rank 6. I could have "fixed" that by
sending ten chunks to the LLM instead of three, but that costs tokens and
triggers lost-in-the-middle, where models under-weight material in the centre
of their context. The goal is **precision at the top**, not more context.

A cross-encoder took hit rate from 87.5% to 100% and MRR from 0.812 to 0.917 at
k=3.

**Follow-up: what makes a cross-encoder different?**

Dense retrieval compares two vectors computed **independently** — the query
never saw the document, the document never saw the query. That is what makes it
fast, because every chunk is embedded once, offline. It is also its ceiling: a
bi-encoder must compress a whole chunk into a single vector *before knowing
what will be asked of it*.

A cross-encoder reads the query and the document **together** in one forward
pass and scores that pair. It cannot be precomputed, so it is far too expensive
across a corpus and much more accurate across a shortlist.

That asymmetry is the whole architecture: cheap-and-broad retrievers narrow
1,309 chunks to 10, then expensive-and-precise ordering over those 10.

---

## Q2. How did you know a reranker was what you needed?

**This is the strongest thread in the project — the answer chains three
modules.**

Module 3 gave a baseline: 87.5% at k=5, 100% at k=10. Recall solved, ranking
not.

Module 4 tried fusion and it did not help. Investigating *why* gave a precise
statement: for the failing query the answer sat at rank 6 in dense and rank 5
in sparse. **Both retrievers agreed it was mediocre, and RRF rewards
agreement** — so no scheme that recombines those two rankings could promote it.

That is a structural limit, not a tuning problem, and it names its own
solution: something that reads the query and document together rather than
recombining independently-formed opinions.

So Module 5 started with a specific, falsifiable prediction — chunk 127 should
move from rank 6 into the top 3 — instead of "let's try a reranker because
everyone uses one." It moved to rank 3.

**Follow-up: how did you verify?**

The audit output, not the aggregate. It prints the matched rank, chunk ID and
excerpt for every pass: chunk 127 at rank 3, excerpt *"The Company uses
derivative instruments, such as foreign currency forward and option
contrac…"*. A hit rate going up is not proof the thing you built is why.

---

## Q3. Tell me about a bug that taught you something.

**Lead with this one.**

My first rerank measurement showed **no improvement at all** — hybrid+rerank
was identical to hybrid. The reranker had never run.

Cohere's trial tier allows about ten requests per minute. An eval run issues
dozens back to back, so nearly every call returned `TooManyRequestsError`. And
I had built the reranker to **fail open**: log a warning, return the
un-reranked candidates. The harness scored the fallback and reported it as the
reranker's number.

The run completed. The numbers looked plausible. The only signal was a warning
whose detail lived in a structured `extra` field the console formatter does not
print.

**The principle: graceful degradation and honest measurement are in direct
conflict.** In a service, failing open is right — reranking is an enhancement,
and failing a user's query over an optional stage is worse than a slightly
worse ranking. In an evaluation it is a trap, because *you cannot measure a
component that is quietly not running*.

The same component needs opposite policies in the two contexts. So `fail_open`
is a constructor argument, the eval CLI passes `False`, and there is a test for
each behaviour.

**Follow-up: what else did you change?**

Two things. Bounded retry with exponential backoff, which any per-request paid
API needs regardless of this bug. And wrapping provider exceptions in
`RetrievalError` — a raw `cohere.TooManyRequestsError` escaping `search()`
broke the contract that catching `SecfilerRagError` catches everything this
package raises.

---

## Q4. What is the index-alignment discipline?

**Answer.** Cohere returns `results[i].index` — a position into the list you
*sent*, not your own identifier. The code maps scores back through that
`index`, never through the result's position in the response.

Cohere does return results sorted by relevance. Relying on that would mean that
if it ever stopped being true, documents and scores would be silently mispaired
— and the output would still look like a perfectly plausible ranking. Nothing
downstream could detect it.

There is a test whose fake client deliberately returns results in reverse-score
order. Code that pairs by response position fails it; code that pairs by
`index` passes.

**Follow-up: why not LangChain's `CohereRerank`?**

Two reasons. It is a `BaseRetriever`, which takes only a query — the same
per-query-filter problem that shaped the whole retrieval layer. And it hides
both the index mapping and the failure policy, which are exactly the two things
in this module worth being explicit about.

---

## Q5. Is hybrid search still worth it?

**Answer.** On my numbers, no — and I say so.

| Strategy | Hit rate @ k=3 | MRR |
|---|---|---|
| dense | 87.5% | 0.812 |
| hybrid | 87.5% | 0.812 |
| dense+rerank | 100.0% | 0.917 |
| hybrid+rerank | 100.0% | 0.917 |

`dense+rerank` and `hybrid+rerank` are **identical**. Once a reranker exists,
fusion adds nothing on this dataset — BM25 is costing a second retriever per
query for zero measured gain.

I have not removed it, because eight eval items cannot support a claim as
strong as "never". But if I had to ship today I would ship **dense +
reranking**, and I would say that rather than keeping a component because it
sounds sophisticated.

**Follow-up: when would BM25 earn its place back?**

Queries with rare exact identifiers — product names, section references, ticker
symbols, specific figures — where an embedding blurs the token into its
neighbourhood. My eval set is mostly natural-language questions, which is
dense's home ground. A dataset with more lexical queries would likely tell a
different story, which is exactly why the conclusion is "unproven here" rather
than "useless".

---

## Q6. What does reranking cost?

**Answer.** Latency roughly doubles: 464 ms to 886 ms median per query. Plus a
per-query paid API call, and a rate limit that is a real operational
constraint.

It does not matter yet. Once generation lands the LLM call is 1–4 seconds and
dominates. And reranking is the rare change that pays for itself: sending 3
precise chunks instead of 10 mediocre ones is **cheaper in tokens, faster to
generate, and more accurate**.

**Follow-up: self-hosted alternative?**

`bge-reranker-large` — no rate limit, no per-query cost, no third party in the
request path. It needs a GPU to be fast and model serving to operate. The right
answer at volume; the wrong one for a project that has to be reproducible from
a clone.

---

## Q7. Walk me through the full retrieval funnel.

```
1,309 chunks
   ├─ dense  top-10 ┐
   │                ├─ RRF (k=60) ─► 10 fused ─► cross-encoder ─► top-3 ─► LLM
   └─ sparse top-10 ┘
```

Each stage is more expensive and more precise than the last, so each one sees
fewer candidates. Retrievers are cheap and imprecise, so they cast wide.
Fusion is free. Reranking costs per document, so it sees ten, not 1,309. The
LLM is most expensive and most context-limited, so it sees three.

**The rule that falls out: never slice to the final `top_k` before fusion or
reranking.** A chunk cut early cannot be recovered by any later stage, however
good. That is not theoretical — the answer to my failing query was at rank 6,
so a funnel that sliced to 5 anywhere upstream would have lost it permanently.

---

## Q8. What is still weak?

1. **Eight eval items.** 100% on eight questions is a smoke test passing, not a
   solved problem. One item is 12.5 points.
2. **Rate limits make evaluation slow** — a full sweep takes minutes of backoff
   on the trial tier.
3. **Hybrid's value is unproven** post-reranking, and it is still in the code.
4. **No caching.** Identical queries re-embed and re-rerank; both are
   deterministic and cacheable.
5. **Still retrieval-only.** Nothing measures whether the answer generated from
   these three chunks is faithful to them — that needs generation and an LLM
   judge.
