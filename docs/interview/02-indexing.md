# Interview — Module 2: Indexing

The write path: embeddings, collection lifecycle, and the identity scheme that
makes re-indexing safe.

---

## Q1. Walk me through indexing.

**Answer.** Three modules. `embeddings` builds the OpenAI model lazily and
fails fast with a named error if the key is missing. `collection` owns the
Qdrant client and the collection lifecycle — create, verify, payload index.
`indexer` derives point IDs, batches, and upserts through LangChain's
`QdrantVectorStore`.

It is deliberately a separate package from `retrieval`, so the read path never
imports write-path code. A read-only service carries no embedding credentials
and no upsert logic.

---

## Q2. How do you handle re-indexing?

**The best question in this module.**

Point IDs are `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")` — deterministic,
so re-running the indexer overwrites in place rather than appending.

That matters because the failure mode of auto-generated IDs is silent: the
second run *succeeds*, the corpus doubles, and you notice weeks later when
every result set contains the same passage three times and the reranker has
nothing diverse to work with.

My first version used the chunk's position as the ID, which collided across
companies — every filing restarts at 0, so Tesla's chunk 0 overwrote Apple's.
Points silently vanished. That is why identity is the **pair**, not the
integer.

**Follow-up: why the separator?**

Without it, `("aapl1", 2)` and `("aapl", 12)` both produce `aapl12`. It costs a
character and removes a collision class.

**Follow-up: why UUID5 rather than UUID4?**

UUID5 is a hash of the name — same input, same output, on any machine, forever.
UUID4 is random, which is exactly what we are avoiding. Qdrant only accepts
UUIDs or unsigned integers as point IDs, so a natural string key has to be
hashed into one.

**Follow-up: what happens if you change the chunk size?**

Every `chunk_id` renumbers, so IDs change and stale points from the old
configuration linger alongside the new ones. That is what `--recreate` is for,
and it is worth remembering *before* tuning the chunker rather than after.

---

## Q3. One collection or one per company?

**Answer.** One collection, `filings`, with a payload filter on
`metadata.company`.

A payload filter is a **soft, query-time boundary**: one query can scope to a
company, widen to several, or drop the filter entirely. Per-company collections
are a hard storage boundary — a cross-company question would mean fanning out
across collections and merging rankings myself.

I would switch if tenants needed genuine isolation, but that is a data-privacy
requirement, not a retrieval one.

**Follow-up: does filtering hurt performance?**

Only without an index. Without one, Qdrant applies the filter *after* the
vector search, so a company-scoped query still pays to search every other
company's vectors — and recall degrades, because the candidate pool it filtered
was shared. With a keyword payload index on `metadata.company`, the filter
narrows the search space up front.

---

## Q4. What surprised you integrating LangChain with Qdrant?

**Answer.** `QdrantVectorStore` does not store your metadata at the payload
root. It nests it:

```json
{"page_content": "...", "metadata": {"company": "aapl", "chunk_id": 12}}
```

So every filter and every payload index must address `metadata.company`.
Filtering on `company` matches nothing — and raises nothing.

I pinned it as a single constant, `COMPANY_PAYLOAD_FIELD`, and wrote a test
asserting its value. That is exactly the kind of framework detail that
silently breaks a filter during a refactor, and a test is cheaper than
rediscovering it.

**Follow-up: so is the abstraction worth it?**

Here, yes — `QdrantVectorStore` gives me `.as_retriever()` for free, which is
what lets the eval harness stay strategy-agnostic in the next module. But I own
the collection creation rather than letting the store auto-create it, because
auto-creation infers the vector size from a probe string and makes the schema a
side effect of whichever code path ran first.

That is the general pattern: **use the abstraction for the interface, own the
schema.**

---

## Q5. Why verify the vector dimension?

**Answer.** Because it is the mismatch that is otherwise invisible. If a
collection was created for 1536 dimensions and you later switch to a model
producing 3072, writes either fail with an opaque server error or — worse in a
named-vector setup — succeed against the wrong configuration, and retrieval
quietly returns nonsense.

`ensure_collection` reads the existing collection's configured size and raises
`IndexingError` naming both numbers and the fix. One clear message at startup
instead of a debugging session at query time.

---

## Q6. Why cosine distance?

**Answer.** OpenAI embeddings encode meaning in *direction*, not magnitude.
Euclidean distance would let a longer document's larger vector norm affect
ranking for reasons unrelated to relevance. Cosine normalises that away.

For already-normalised vectors, cosine and dot product rank identically — dot
is marginally faster. Cosine is the safe default because it does not depend on
the model continuing to return normalised vectors.

---

## Q7. How do you test this without Docker or an API key?

**Answer.** Two techniques.

`QdrantClient(location=":memory:")` runs the **real** local Qdrant engine
in-process — real collection config, real upserts, real payload filters — with
no container and no network. Combined with
`DeterministicFakeEmbedding`, that makes the whole write path testable in
milliseconds: idempotency, metadata round-trip, batching, filter scoping, and
the dimension-mismatch error.

These are genuine behaviour tests, not mocks. A mocked Qdrant would only prove
I called the methods I think I called.

**What in-memory mode cannot prove** is also worth naming: payload indexes have
no effect locally, so "the index actually narrows the search" is covered by an
integration test against a live server, along with the fact that the real
embedding model returns exactly 1536 dimensions.

**Follow-up: why not just mock the vector store?**

Because the bugs in this module live in the *interaction* — that metadata is
nested, that duplicate IDs upsert rather than append, that a dimension mismatch
raises. A mock asserts my assumptions back to me. The in-memory engine tests
the thing.

---

## Q8. What is the cost, and what would break at scale?

**Cost.** 1,309 chunks at ~250 tokens each is roughly 330K tokens — well under
a cent with `text-embedding-3-small`, once, at index time. Query embeddings are
one call per question. Embedding cost is not the constraint here; generation is.

**What breaks first at scale:**

1. **Batch size and rate limits.** Batching is at 100 chunks per request so one
   failure costs a batch, not the corpus. At millions of chunks this needs
   backoff and concurrency control.
2. **No resume.** A failed run restarts from the beginning. Deterministic IDs
   make that *safe* — already-written points are simply overwritten — but not
   *fast*. A checkpoint would fix it.
3. **Single-node Qdrant.** Sharding, quantisation and HNSW tuning arrive well
   before this corpus does.

---

## Q9. What is still weak here?

1. **No resume/checkpoint** on a partially failed run.
2. **No orphan cleanup** — if the chunker produces fewer chunks than a previous
   run, the extra points from that run stay behind. `--recreate` is the blunt
   fix; a proper one would delete points whose `chunk_id` exceeds the new count.
3. **Sync client only.** Correct for a batch job, but the read path will need
   `AsyncQdrantClient` — a sync client inside `async def` blocks the event loop.
4. **No sparse vectors yet.** Qdrant supports them natively, which is likely a
   better home for BM25 than an in-memory index. That decision lands with the
   hybrid module.
