# Performance and Scaling

Current scale: **3 filings, ~800 chunks, one user.** Everything below is
written as *what changes when* — because "we would use Kubernetes" is not an
answer, but "in-memory BM25 breaks at N documents, here is the threshold and
the replacement" is.

---

## 1. Where the time actually goes

Per query, order of magnitude:

| Stage | Cost | Share |
|---|---|---|
| Query embedding | 50–150 ms | small |
| Qdrant ANN + filter | 5–30 ms | negligible |
| BM25 (in-memory, ~800 docs) | 5–50 ms | small |
| RRF fusion | <1 ms | none |
| Cross-encoder rerank (10 docs) | 100–400 ms | moderate |
| **LLM generation** | **1–4 s** | **dominant** |

**Consequence:** retrieval micro-optimisation is invisible to users. The
leverage is in generation — streaming, smaller models for simple questions,
caching — and in *not calling the LLM twice*.

---

## 2. Cost model

| Operation | Cost driver | Note |
|---|---|---|
| Indexing | One embedding call per chunk, once | ~800 chunks is cents; re-indexing on every deploy is the waste |
| Query embedding | One per query | Cacheable — identical queries repeat more than you expect |
| Reranking | Per candidate document | Why the funnel is 10 → 3, not 100 → 3 |
| Generation | Input + output tokens | Sending 10 chunks instead of 3 roughly triples input cost *and* hurts accuracy (lost-in-the-middle) |

Sending fewer, better chunks is the rare choice that improves quality **and**
latency **and** cost simultaneously. That is why reranking pays for itself.

---

## 3. Scaling dimensions and their breaking points

### 3.1 More documents

| Scale | What breaks | Response |
|---|---|---|
| ~800 chunks (now) | Nothing | In-memory BM25 is fine |
| ~10⁴–10⁵ | BM25 memory + startup rebuild time | Move sparse search into Qdrant (native sparse vectors) or OpenSearch |
| ~10⁶+ | Flat/naive ANN latency, single-node memory | HNSW tuning, quantisation, sharding, payload indexes on `company` |

**The in-memory BM25 index is the first thing to break.** It is rebuilt at
process start from the whole corpus — fine at 800 chunks, a multi-minute
startup at a million. The fix is not a bigger machine; it is moving sparse
retrieval into the store that already holds the data.

### 3.2 More queries per second

The read path is I/O-bound (three network calls), so:

- **Async everywhere on the read path.** A sync client in an `async` handler
  blocks the event loop and turns a latency problem into an outage
- Connection pooling / a shared client, not per-request construction
- Horizontal scaling: the API is stateless *except* for the BM25 index, which
  is why moving it into Qdrant also unlocks clean replicas
- Bounded concurrency to upstream APIs so a traffic spike does not turn into a
  rate-limit storm

### 3.3 More companies / tenants

Single collection + payload filter (current) scales well *if* `company` has a
payload index — without it, Qdrant filters post-hoc and the "cheap filter"
becomes a full scan.

Per-tenant collections become right when tenants need hard isolation,
independent lifecycle, or wildly different corpus sizes. That is a data-privacy
decision more than a performance one.

### 3.4 Bigger index, same machine

- **Quantisation** (scalar/binary) cuts memory several-fold for a small recall
  loss — measure the loss, do not assume it
- **HNSW `m` / `ef_construct`**: higher = better recall, more memory and slower
  builds
- **Payload indexes** on every filtered field

---

## 4. Caching — in order of value

| Layer | Key | Hit rate | Risk |
|---|---|---|---|
| Query embedding | Normalised query text | Moderate | None — deterministic |
| Retrieval results | (query, filters, k) | Moderate | Stale after re-index — version the key with an index generation |
| Full answer | (query, filters, prompt version, model) | Lower | Stale prompts/models — the key must include both |

Cache invalidation is the whole difficulty: **a cache key that does not include
the index version will serve pre-re-index answers forever.** If you take one
thing from this section, take that.

---

## 5. Reliability

| Dependency | Failure | Behaviour |
|---|---|---|
| Qdrant | Down | Fail fast, `503`. There is no answer without retrieval |
| Embeddings | Rate limited | Retry with backoff; surface `429` |
| Reranker | Down/slow | **Degrade** — skip reranking, log it, still answer |
| LLM | Down | `503`; optionally return retrieved chunks so the user sees sources |

The reranker is the only optional stage. Naming which dependencies are
load-bearing and which are enhancements is a design decision worth stating
explicitly.

---

## 6. What I would do first, at 100× scale

1. Move BM25 into Qdrant sparse vectors (removes the only stateful startup cost)
2. Add a payload index on `company`
3. Cache query embeddings, keyed with an index generation
4. Stream generation so time-to-first-token drops even if total time does not
5. Only then consider replicas and quantisation — after measuring, not before

The ordering is the answer to the interview question. Anyone can list
techniques; the signal is knowing which constraint binds first.
