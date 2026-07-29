# Debugging Guide

RAG fails *quietly*. The system returns a fluent, confident, wrong answer and
nothing raises. This guide is ordered by the question people actually ask —
"why is the answer wrong?" — and works backwards through the pipeline.

---

## 1. First move: bisect retrieval vs generation

Before touching prompts or models, answer one question:

> **Was the correct chunk in the context the LLM received?**

```python
docs = retriever.invoke("what was Apple's total revenue?")
for d in docs:
    print(d.metadata["company"], d.metadata["chunk_id"], d.page_content[:300])
```

| Correct chunk present? | The bug is in… | Go to |
|---|---|---|
| No | Retrieval (or ingestion before it) | §2 |
| Yes, but the answer is wrong | Generation | §3 |

Skipping this step is why people spend a day tuning prompts for a retrieval
bug. Half of all "hallucination" reports are retrieval misses.

---

## 2. Retrieval debugging

### 2.1 Is the chunk even in the index?

```bash
curl -s http://localhost:6333/collections/filings | jq '.result.points_count'
```

Then scroll for it by payload filter rather than trusting a search:

```python
client.scroll(
    collection_name="filings",
    scroll_filter=Filter(must=[FieldCondition(key="company", match=MatchValue(value="aapl"))]),
    limit=3,
)
```

Common causes of "it is not there":
- Indexing ran against a different `DATA_DIR`
- The collection was recreated after indexing
- A filter mismatch — see 2.2

### 2.2 Casing and filter mismatches (silent, common)

Payload filters are exact matches. `"AAPL"` does not match `"aapl"`. The
symptom is a perfectly working system returning zero results with no error at
all.

**Convention: company keys are lowercase everywhere** — filenames, metadata,
eval data, filter values. When results come back empty, drop the filter first:
if results appear, the bug is the filter, not the search.

### 2.3 Is it a chunking problem?

Print the chunk that *should* have matched:

- Is the fact split across two chunks? → increase overlap, or chunk on
  structure rather than fixed size
- Is the chunk mostly boilerplate with one relevant sentence? → the embedding
  is diluted; chunks that are too large retrieve poorly
- Are the numbers missing (`Products $ $ $`)? → the HTML table did not survive
  cleaning. This is a known failure mode of naive text extraction, and it is an
  *ingestion* bug, not a retrieval one.

### 2.4 Dense vs sparse disagreement

Run both retrievers separately on the failing query.

| Dense finds it | Sparse finds it | Diagnosis |
|---|---|---|
| ✅ | ❌ | Vocabulary mismatch — user words differ from filing words. Expected; this is why hybrid exists |
| ❌ | ✅ | Embedding/semantic miss — often an over-long or boilerplate-heavy chunk |
| ❌ | ❌ | Not in the index, wrong filter, or bad chunking. Go to 2.1 |
| ✅ | ✅ | The bug is in fusion, reranking, or the final `top_k` slice |

### 2.5 Lost between fusion and the LLM

Log the ranked list at each stage: after each retriever, after RRF, after
rerank. The failure is nearly always the same shape — **the correct chunk was
present at rank 5 and the pipeline sliced to 3 before the reranker ran.**

---

## 3. Generation debugging

The correct chunk was in context and the answer is still wrong.

| Symptom | Likely cause | Fix |
|---|---|---|
| Invents a number not in context | Prompt does not forbid outside knowledge | Explicit grounding instruction + refusal path |
| "I don't know" with good context | Over-strict prompt, or context buried mid-prompt | Reorder; put context and question close together |
| Ignores later chunks | Lost-in-the-middle: models attend to the start and end | Fewer, better chunks — this is what reranking buys |
| Cites the wrong chunk | Citation markers not tied to chunk metadata | Number the context blocks and require the model to cite those numbers |
| Mangled `$` figures | Ingestion lost the table | Fix in ingestion; no prompt repairs missing data |

**Always print the fully rendered prompt** at least once per prompt change. What
you think you sent and what the template produced are different things more
often than anyone admits.

---

## 4. Infrastructure symptoms

| Symptom | Check | Fix |
|---|---|---|
| `Connection refused :6333` | `docker compose ps` | `docker compose up -d` |
| `Vector dimension error` on upsert | Collection dim vs model dim | Recreate the collection; 1536 for `text-embedding-3-small` |
| `429` from OpenAI | Batch size / rate | Batch embeddings, add backoff |
| Latency spikes under concurrency | Sync client in an `async def` | Use `AsyncQdrantClient` on the read path |
| Every log line appears 3× | `configure_logging` called repeatedly | It is idempotent by design — check for a stray `basicConfig` |
| Config change has no effect | Cached settings singleton | `get_settings.cache_clear()`, or restart |

---

## 5. Tools

**Turn up our logs without third-party noise:**

```bash
LOG_LEVEL=DEBUG uv run python -m secfiler_rag...
```

`configure_logging` keeps third-party loggers at WARNING deliberately — `DEBUG`
on `httpx` buries your own output.

**LangSmith** (once Module 7 lands) is the highest-leverage tool here: it shows
the actual retrieved documents and the actual rendered prompt for a given
`run_id`, which collapses most of §2 and §3 into one page.

**Qdrant dashboard**: <http://localhost:6333/dashboard> — inspect points and
payloads directly.

---

## 6. The audit habit

When a metric improves, read *which* items passed and *why*.

A pass can be an accident: a loose expected-substring like `"net sales"` matches
dozens of chunks, so the harness reports a hit while retrieval actually failed.
Two such false positives were caught in the previous build. **A green number you
have not audited is not evidence.**
