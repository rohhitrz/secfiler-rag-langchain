# Low-Level Design (LLD)

This document grows **with the code**. Each module section is filled when that module lands. Until then, the section describes the **intended contract** so implementation stays aligned.

For package placement, see [`03-folder-structure.md`](03-folder-structure.md).

---

## Shared conventions

| Concern | Rule |
|---|---|
| Company identifiers | Lowercase: `aapl`, `msft`, `tsla` |
| Chunk identity | Integer `chunk_id` per company, stable for a given chunker config |
| Point identity (Qdrant) | `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")` |
| Document type | `langchain_core.documents.Document` |
| Config access | Only via `get_settings()` |
| Logging | `get_logger(__name__)` + `extra={...}` for structured fields |
| Errors | Raise subclasses of `SecfilerRagError` |

---

## Module 0 — Foundation (implemented)

### `config.settings.Settings`

- pydantic-settings `BaseSettings`, `frozen=True`, `extra="ignore"`
- Cached singleton: `get_settings()`; tests call `cache_clear()`
- Secrets as `SecretStr`

### `core.logging`

- Stdlib logging; console or JSON
- Idempotent `configure_logging`
- Project logger `secfiler_rag.*` at configured level; third parties at WARNING

### `core.exceptions`

```
SecfilerRagError
├── ConfigurationError
├── IngestionError
├── IndexingError
├── RetrievalError
└── EvaluationError
```

Leaf types are added as each module lands, so the hierarchy stays a truthful
map of how this system actually fails.

---

## Module 1 — Ingestion (implemented)

Four modules, each independently testable. Only `pipeline` touches both the
filesystem and configuration, which is what keeps the other three reusable from
a test or a notebook.

```text
loader.py    parse_filing_name(Path) -> FilingSource
             discover_filings(Path)  -> list[FilingSource]
             read_filing(FilingSource) -> str

cleaner.py   clean_html(str) -> str

splitter.py  split_filing(text, *, company, source, chunk_size, chunk_overlap)
                 -> list[Document]

pipeline.py  ingest_filing(FilingSource, *, chunk_size=None, chunk_overlap=None)
                 -> list[Document]
             ingest_all(raw_dir=None, *, chunk_size=None, chunk_overlap=None)
                 -> list[Document]
```

### `FilingSource`

Frozen dataclass: `path`, `company`, `fiscal_year`, plus a `source` property
returning the filename.

### Filename convention (enforced, not inferred)

```
^(?P<company>[a-z][a-z0-9]{0,9})-(?P<year>\d{4})\.html?$
```

`aapl-2025.htm` → `company="aapl"`, `fiscal_year=2025`.

Lowercase-only is deliberate. `AAPL-2025.htm` raises rather than producing an
uppercase company key that would never match a lowercase Qdrant payload
filter — a mismatch that returns zero results with no error at all.

### Cleaning contract

| Input | Treatment |
|---|---|
| `script`, `style`, `head`, `noscript` | `decompose()` |
| `ix:header`, `ix:hidden`, `ix:references`, `ix:resources` | `decompose()` — machine-only |
| Any other `ix:*` | **`unwrap()`** — the tag goes, the value stays |
| `<table>` | One line per `<tr>`, cells joined ` \| `, innermost table first |
| Empty table cells | Dropped (visual spacers) |
| Orphaned `$` / `%` cells | Reattached: `$ \| 416,161 \| 6 \| %` → `$416,161 \| 6%` |
| Whitespace | Collapsed *within* lines; newlines preserved |

Rationale in [ADR 0007](adr/0007-preserve-inline-xbrl-values.md).

### Metadata contract on every `Document`

| Key | Type | Meaning |
|---|---|---|
| `company` | `str` | `aapl` / `msft` / `tsla` |
| `chunk_id` | `int` | Position in that company's chunk list, from 0 |
| `source` | `str` | Source filename |
| `start_index` | `int` | Character offset in the cleaned text |

**Identity is the pair `(company, chunk_id)`.** IDs restart per filing, so
`chunk_id` alone collides across companies.

### Chunking

`RecursiveCharacterTextSplitter`, separators `["\n\n", "\n", ". ", " ", ""]`,
size/overlap from `Settings` (1000/200), `add_start_index=True`. See
[ADR 0008](adr/0008-recursive-chunking-strategy.md).

### Corpus results (baseline, 1000/200)

| Company | Raw | Clean text | Chunks |
|---|---|---|---|
| aapl | 1.5 MB | 209,393 chars | 292 |
| msft | 8.2 MB | 317,163 chars | 441 |
| tsla | 2.4 MB | 399,145 chars | 576 |
| **Total** | **12.1 MB** | | **1,309** |

Previous build: 768 chunks. The +70% is recovered content, not smaller chunks.

### Failure modes

| Condition | Result |
|---|---|
| Filename violates the convention | `IngestionError` |
| Raw directory missing or empty | `IngestionError` |
| File unreadable or empty | `IngestionError` |
| Cleaning yields no text | `IngestionError` |
| Splitting yields no chunks | `IngestionError` |
| `chunk_overlap >= chunk_size` | `ValidationError` at startup |
| Undecodable bytes | Replaced, not raised — one lost character beats one lost filing |

`ingest_all` fails the whole run if any filing fails. Partial ingestion is
worse than a loud failure: a missing filing becomes a retrieval gap that
surfaces weeks later as an unexplained bad answer.

---

## Module 2 — Indexing (implemented)

```text
embeddings.py  build_embeddings(settings=None) -> Embeddings

collection.py  build_client(settings=None) -> QdrantClient
               ensure_collection(client, *, collection_name, vector_size,
                                 recreate=False) -> bool   # True if created
               ensure_payload_index(client, *, collection_name) -> None
               count_points(client, collection_name) -> int

indexer.py     point_id(company, chunk_id) -> str
               build_vector_store(client, embeddings, *, collection_name)
                   -> QdrantVectorStore
               index_documents(documents, *, client, embeddings,
                               settings=None, recreate=False) -> int
```

CLI: `scripts/index_filings.py [--recreate] [--dry-run]`

### Collection schema

| Property | Value |
|---|---|
| Name | `filings` (`QDRANT_COLLECTION`) |
| Vector size | `EMBEDDING_DIMENSIONS` (1536) |
| Distance | Cosine — OpenAI embeddings encode meaning in direction, not magnitude |
| Payload index | `metadata.company`, keyword |

Created explicitly, never auto-created on first write. An existing collection's
vector size is **verified**; a mismatch raises `IndexingError` naming both
numbers.

### Payload layout (LangChain-imposed)

```json
{
  "page_content": "Total net sales | $416,161 | 6% | ...",
  "metadata": {"company": "aapl", "chunk_id": 12,
               "source": "aapl-2025.htm", "start_index": 9840}
}
```

`QdrantVectorStore` nests metadata rather than storing it at the root, so
**every filter and index addresses `metadata.company`**, exposed as the single
constant `COMPANY_PAYLOAD_FIELD`. A filter on the bare field name matches
nothing and raises nothing.

### Point identity

```python
point_id = str(uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}"))
```

Deterministic → re-indexing upserts in place. `company` is required because
`chunk_id` restarts per filing; the `-` separator prevents
`("aapl1", 2)` colliding with `("aapl", 12)`. Rationale in
[ADR 0009](adr/0009-deterministic-point-ids.md).

### Failure modes

| Condition | Result |
|---|---|
| `OPENAI_API_KEY` missing | `ConfigurationError` naming the variable |
| Empty document list | `IndexingError` |
| Document missing `company` / `chunk_id` | `IndexingError` |
| Two documents share `(company, chunk_id)` | `IndexingError` — identity broken upstream |
| Existing collection has a different vector size | `IndexingError` with both sizes and the fix |
| Payload index already exists | No-op |

### Testing approach

Unit tests use `QdrantClient(location=":memory:")` — the real local engine, no
Docker — with `DeterministicFakeEmbedding`. That covers idempotency, metadata
round-trip, batching, filter scoping and the dimension check in milliseconds.

Payload indexes have no effect in local mode, so index behaviour and the real
1536-dimension embedding output are covered by integration tests against a live
server (`uv run pytest -m integration`).

---

## Module 3 — Dense retrieval + evaluation (implemented)

```text
retrieval/dense.py     DenseRetriever(store, *, default_top_k=5)
                         .search(query, filters=None, top_k=None) -> list[Document]
                         .as_search_fn() -> SearchFn
                         .as_langchain_retriever(*, filters=None, top_k=None)
                       build_filter(filters) -> qmodels.Filter | None

evaluation/dataset.py  load_dataset(path) -> EvalDataset
                       EvalItem(query, expected_substring, filters, tier)

evaluation/harness.py  evaluate(dataset, search_fn, *, top_k=5) -> EvalReport
                       SearchFn = Callable[[str, Mapping[str, Any], int],
                                           Sequence[Document]]

evaluation/metrics.py  hit_rate(hits) -> float
                       mean_reciprocal_rank(ranks) -> float
```

CLI: `scripts/evaluate_retrieval.py [--in-memory] [--top-k K...] [--audit]`

### The harness contract (frozen)

The harness receives a **callable**, never a retriever object, and forwards
`filters` without inspecting them. It must never learn that companies, BM25 or
Qdrant exist. Enforced by `test_harness_never_inspects_filters`. Rationale in
[ADR 0010](adr/0010-retriever-agnostic-eval-harness.md).

Knowledge is split so that stays true:

| Component | Knows |
|---|---|
| Dataset loader | The dataset's on-disk schema |
| Harness | That items have queries, expected text, and opaque filters |
| `DenseRetriever` | That `company` maps to `metadata.company` |

### Filter translation

`{"company": "aapl"}` → `Filter(must=[FieldCondition(key="metadata.company",
match=MatchValue(value="aapl"))])`. An unknown key raises `RetrievalError` — a
silently dropped filter returns another company's chunks and looks like a
quality problem rather than a bug.

### Report surface

| Member | Purpose |
|---|---|
| `hit_rate` | Fraction of items whose expected text was retrieved |
| `mrr` | Mean reciprocal rank — position, not just presence |
| `median_latency_ms` | Retrieval latency, harness overhead excluded |
| `by_tier(n)` | Tier 1 (smoke) and tier 2 (realistic) scored separately |
| `misses` | Failed items with their top results, for diagnosis |
| `ItemResult.matched_rank / matched_chunk_id / matched_excerpt()` | Auditing a pass |

### Baseline (dense only, 1,309 chunks, `text-embedding-3-small`)

| k | Hit rate | MRR |
|---|---|---|
| 1 | 75.0% | 0.750 |
| 3 | 87.5% | 0.812 |
| 5 | 87.5% | 0.812 |
| 10 | **100.0%** | 0.833 |

Recall is solved by k=10; MRR moves only 0.812 → 0.833. **The gap is precision
at the top, not recall** — a reranker's job, now measured rather than assumed.

### Failure modes

| Condition | Result |
|---|---|
| Empty query | `RetrievalError` |
| Unknown filter key | `RetrievalError` listing supported keys |
| Filter matches nothing | Empty list — a valid outcome, not an error |
| Dataset missing / malformed / empty | `EvaluationError` |
| Item without query or expected substring | `EvaluationError` — never skipped, since a dropped item makes the denominator wrong |

---

## Module 4 — Sparse + hybrid retrieval (implemented)

```text
filters.py  FILTER_FIELDS: dict[str, str]        # key -> Qdrant payload path
            validate_filters(filters) -> None
            build_qdrant_filter(filters) -> Filter | None   # dense, server-side
            matches(metadata, filters) -> bool              # sparse, in-Python

sparse.py   tokenize(text) -> list[str]
            SparseRetriever(documents, *, default_top_k=5)
              .search(query, filters=None, top_k=None) -> list[Document]

fusion.py   reciprocal_rank_fusion(result_lists, *, k=60, top_k=10)
                -> list[Document]

hybrid.py   Retriever  (Protocol: .search(query, filters, top_k))
            HybridRetriever(retrievers, *, candidate_k=10, rrf_k=60,
                            default_top_k=5)
```

### Shared filter vocabulary

Dense pushes filters into Qdrant; sparse applies them in Python. Same meaning,
two mechanisms — so the vocabulary lives in `filters.py` and each strategy
translates. Without that, one strategy could honour a key the other silently
ignored, and a hybrid query would return one company's chunks from one side and
everyone's from the other.

Note the asymmetry: Qdrant filters use the **prefixed** path
(`metadata.company`) because payloads are nested; in-memory `Document`s hold
metadata **flat**, so `matches()` compares the bare key.

### Tokenizer

`re.findall(r"[a-z0-9]+", text.lower())` — one function, used for **both**
corpus indexing and queries. Asymmetry degrades matching silently. No stemming
or stop words: IDF already discounts ubiquitous terms.

### RRF

```
score(doc) = Σ over retrievers  1 / (k + rank),   k = 60
```

- Identity: `(company, chunk_id)`; falls back to page content when metadata is
  absent, so fusion still de-duplicates
- Ties break on the identity key, so fusion is deterministic and symmetric in
  retriever order
- `score` is overwritten with the RRF score; `rrf_ranks` records the
  per-retriever ranks that produced it

### The funnel

```
dense  top-10 ┐
              ├── RRF ──► top-10 fused ──► (Module 5: rerank) ──► top-3
sparse top-10 ┘
```

`candidate_k` is always at least the requested `top_k`. **Never slice to the
final `top_k` before fusion or reranking** — a chunk cut early cannot be
recovered downstream.

### Measured comparison (same index, same harness, same dataset)

| Strategy | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| dense | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |
| sparse | 50.0% / 0.500 | 75.0% / 0.604 | 87.5% / 0.629 | 100% / 0.647 |
| hybrid | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |

Hybrid did **not** improve on dense. Verified not a bug (see
[ADR 0012](adr/0012-rrf-hybrid-fusion.md)): fusion reorders genuinely, but on
the one failing query both retrievers rank the answer mid-pack (5 and 6) and
RRF rewards agreement. That is the measured case for a cross-encoder.

### Failure modes

| Condition | Result |
|---|---|
| Empty BM25 corpus | `RetrievalError` |
| Empty query | `RetrievalError` |
| Unknown filter key (any strategy) | `RetrievalError` |
| Query sharing no term with the corpus | Empty list |
| RRF `k <= 0` | `RetrievalError` |
| One retriever returns nothing | Fusion proceeds with the rest |

---

## Module 5 — Cross-encoder reranking (implemented)

```text
rerank.py  RerankResult / RerankResponse / RerankClient   (Protocols)
           RerankingRetriever(base, client, *, model="rerank-v3.5",
                              candidate_k=10, default_top_k=3,
                              fail_open=True, max_retries=5,
                              backoff_seconds=2.0, sleep=time.sleep)
             .search(query, filters=None, top_k=None) -> list[Document]
           build_rerank_client(settings=None) -> RerankClient
```

### Index alignment (the discipline)

Scores map back through `result.index` — a position into the list **sent** —
never through the result's position in the response. Cohere returns results
sorted by relevance, but relying on that would silently mispair documents and
scores if it changed, and the output would still look like a plausible ranking.
Enforced by a test whose fake returns results in reverse-score order.

### Metadata added

| Key | Meaning |
|---|---|
| `score` | Cross-encoder relevance, overwriting the retriever's score |
| `rank_before_rerank` | 1-based position before reranking, for auditing |

### Failure policy — opposite in the two contexts

| Context | `fail_open` | Why |
|---|---|---|
| Service | `True` | Reranking is an enhancement; failing a query over it is worse than a weaker ordering |
| Evaluation | **`False`** | A silently absent reranker gets *measured as if it ran* |

Provider exceptions are wrapped in `RetrievalError`, so catching
`SecfilerRagError` genuinely covers every failure this package raises.

Retryable failures (rate limits, timeouts, 503) get bounded retry with
exponential backoff — 5 retries at a 2 s base is ~62 s cumulative, enough to
clear a ten-per-minute trial limit.

### The full funnel

```
1,309 chunks
   ├─ dense  top-10 ┐
   │                ├─ RRF (k=60) → 10 → cross-encoder → top-3 → LLM
   └─ sparse top-10 ┘
```

### Measured (same index, same harness, top_k=3)

| Strategy | Hit rate | MRR | Median latency |
|---|---|---|---|
| dense | 87.5% | 0.812 | 464 ms |
| hybrid | 87.5% | 0.812 | 497 ms |
| **dense+rerank** | **100.0%** | **0.917** | 886 ms |
| **hybrid+rerank** | **100.0%** | **0.917** | — |

Audited, not assumed: chunk 127 moved rank 6 → **rank 3**. Tier 1 80% → 100%.

`dense+rerank` equals `hybrid+rerank`, so **fusion adds nothing once reranking
exists** on this dataset. See [ADR 0013](adr/0013-cross-encoder-reranking.md).

---

## Modules 6+ — Generation

Contracts filled when those modules land. Standing rules:

- Every retrieval strategy exposes the same `SearchFn` shape
- Every stage that reorders results overwrites `metadata["score"]` with its own
- Generation always receives an explicit list of context documents

---

## Dependency rule (enforced by design)

```text
ingestion  →  indexing  →  retrieval  →  generation
                  ↑              ↑
            evaluation     observability
config / core ← used by all; import nothing from stages
```

`core` must never import `ingestion` / `indexing` / `retrieval` / `generation`.
