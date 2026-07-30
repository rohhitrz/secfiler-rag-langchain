# PROGRESS.md — secfiler-rag (LangChain)

> Source of truth for project state across sessions. Update at the end of every meaningful work session.

---

## Project at a glance

| | |
|---|---|
| **Goal** | Production-grade Advanced RAG over SEC 10-K filings (AAPL, MSFT, TSLA), built on LangChain + Qdrant + LangSmith. Interview-ready, open-source quality. |
| **Repo** | `secfiler-rag-langchain` — LangChain rewrite of the earlier raw-Python `secfiler-rag` build |
| **Corpus** | `data/raw/{aapl,msft,tsla}-2025.htm` (tracked in git) |
| **Stack** | Python 3.12+ · uv · LangChain · LangSmith · Qdrant · OpenAI embeddings · pytest · pydantic-settings |

---

## Syllabus (canonical — do not renumber)

| # | Module | Status |
|---|---|---|
| 0 | Repo cleanup, structure, foundational docs | ✅ **DONE** |
| 1 | Ingestion — HTML load, clean, chunk → `Document` | ✅ **DONE** |
| 2 | Indexing — OpenAI embeddings + Qdrant upsert | ✅ **DONE** |
| 3 | Dense retrieval + company filter + eval harness | ✅ **DONE** |
| 4 | Sparse (BM25) + hybrid RRF fusion | ✅ **DONE** |
| 5 | Cross-encoder reranking | ✅ **DONE** |
| 6 | Answer generation (cited answers) | 🔜 **NEXT** |
| 7 | LangSmith observability + richer evals | Not started |
| 8 | Agentic RAG (routing, multi-step) | Not started |
| 9 | FastAPI serving layer | Not started |
| 10 | Guardrails | Not started |
| 11 | Frontend (optional differentiator) | Not started |
| 12 | Deployment + caching | Not started |

---

## Locked decisions (carry forward + new)

| Decision | Choice | Why |
|---|---|---|
| Corpus | AAPL + MSFT + TSLA FY2025 10-Ks | Real, messy, sellable |
| Filings in git | **Tracked** under `data/raw/` | Clone reproducibility; ~12MB is acceptable |
| Orchestration | LangChain (LCEL / interfaces) | Industry-standard; interview signal. Core RAG design stays ours |
| Vector DB | Qdrant (`qdrant/qdrant:v1.18.2`) via Docker | Hybrid-native, metadata filters, production-grade |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) | Cheap, strong baseline |
| Config | pydantic-settings, no `os.environ` elsewhere | Fail-fast, typed, testable |
| Layout | `src/` package + stage packages | One-way pipeline; acyclic imports |
| Company keys | lowercase (`aapl`, `msft`, `tsla`) | Avoid silent filter mismatches |
| Collection | Single `filings` + payload filter | Soft query-time boundary beats per-company collections |
| Point IDs | Deterministic `uuid5` on `{company}-{chunk_id}` | Idempotent re-index |
| Fusion | RRF (k=60), not score blending | Sidesteps BM25 vs cosine scale mismatch |
| Eval harness | Retriever-agnostic | Honest A/B; no domain knowledge in harness |
| Dependencies | Add only when the module lands | Keeps the graph honest |
| Inline XBRL | **`unwrap()`, never `decompose()`** | The tags wrap the visible figures; decomposing deleted 53% of digits |
| Tables | One line per `<tr>`, cells joined ` \| ` | Keeps a row's label adjacent to its numbers |
| Whitespace | Collapse within lines, keep newlines | A recursive splitter needs boundaries to split on |
| Chunking | `RecursiveCharacterTextSplitter`, 1000/200 | Structure-first; baseline carried over pending eval numbers |
| Chunk identity | `(company, chunk_id)` pair | IDs restart per filing |
| Filenames | `{company}-{year}.htm`, lowercase, enforced | Uppercase would silently never match a payload filter |
| Point IDs | `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")` | Idempotent re-index; the pair is the identity, separator is load-bearing |
| Collection | Single `filings`, explicit creation, dim verified | Auto-creation makes the schema a side effect of whichever path ran first |
| Payload path | **`metadata.company`**, not `company` | `QdrantVectorStore` nests metadata; a bare-field filter matches nothing, silently |
| Distance | Cosine | OpenAI embeddings encode meaning in direction, not magnitude |
| Test strategy | In-memory Qdrant + `DeterministicFakeEmbedding` | Real engine behaviour with no Docker and no API key |
| Eval harness | Takes a `SearchFn`, not a retriever; filters opaque | **FROZEN** — the moment it learns a strategy, cross-strategy numbers die |
| Ground truth | Substring, not chunk_id | Survives re-chunking, so before/after chunker changes stay comparable |
| Metrics | Hit rate **and** MRR, tiers reported separately | A reranker moves MRR while hit rate stays flat |
| Audit | Every pass records rank + chunk + excerpt | A green number nobody read is not evidence |
| Tokenizer | One `tokenize()` for **both** BM25 index and query | Asymmetry degrades matching silently, with no error |
| BM25 index | Single corpus-wide, filtered after scoring | Keeps scores comparable filtered vs unfiltered; per-company IDF parked |
| Fusion | RRF `Σ 1/(60+rank)`, ties broken on identity key | Rank is scale-free; tie-break makes fusion deterministic + symmetric |
| Candidate width | Each retriever 10 → fuse 10 → slice last | The answer sat at rank 6; slicing early loses it irrecoverably |
| Reranker | Cohere `rerank-v3.5` behind a `RerankClient` Protocol | Swappable; tests inject a fake, no provider import in the retriever |
| Index alignment | Map `result.index` back into the sent list | Pairing by response order silently mispairs docs and scores |
| Failure policy | `fail_open=True` in a service, **`False` when measuring** | You cannot measure a component that is quietly not running |
| Rate limits | Bounded retry + exponential backoff (5 × 2s base) | Cohere trial tier is ~10 req/min; an eval issues dozens |

---

## What's built (Module 0)

**Removed:** old `src/secfiler_rag/rag/*` implementation, `main.py`, `config.py`,
`scratch_bm25.py`, `*_clean.txt`, `qdrant_storage/`, all `__pycache__`.
Recoverable from commit `7019a96` and the sibling `secfiler-rag` repo.

**Built:**

- Package layout: `config`, `core`, `ingestion`, `indexing`, `retrieval`, `generation`, `evaluation`, `observability` — each `__init__.py` documents its single responsibility
- Typed `Settings` (frozen, `SecretStr`, `extra="ignore"`) + cached `get_settings()` — no import-time side effects
- Stdlib logging: console + JSON formatters, idempotent setup, third parties pinned at WARNING
- Exception hierarchy (`SecfilerRagError` → `ConfigurationError` / `IngestionError`)
- 16 unit tests; `ruff` + `mypy --strict` clean
- Tooling consolidated in `pyproject.toml` (pytest, ruff, mypy)
- Docs: `docs/README.md` index, HLD, LLD, folder structure, data flow, request lifecycle, setup, development, debugging, failure modes, scaling — plus ADRs 0001–0006 and `interview/00-foundation.md`
- Eval seed set preserved as `evals/datasets/seed_eval_set.json` (+ `evals/README.md`)
- Raw filings now **tracked** in git (ADR 0006)

**Verification:** `uv run pytest` → 16 passed · `uv run ruff check src tests` → clean · `uv run mypy` → clean (18 files)

**Open:** `origin` still points at `git@github.com:rohhitrz/secfiler-rag.git` (the
old project's remote). Repoint before any push. Nothing has been committed or
pushed for Module 0 yet.

---

## What's built (Module 1 — Ingestion)

`ingestion/` = `loader` + `cleaner` + `splitter` + `pipeline`. Only `pipeline`
touches both the filesystem and config, so the middle two stay pure functions.

**The bug that mattered.** The previous build stripped inline XBRL with
`decompose()`, which removes a tag *and its contents* — but `ix:` tags wrap the
visible values. Measured on Apple FY2025: 251 spans of real content deleted,
including `Apple Inc.`, `10-K`, the fiscal year end, and **53% of every digit**.
This was the real cause of the `Products $ $ $` symptom, misfiled as a table
problem for four modules. Fix: `unwrap()`, keeping only the four machine-only
containers on `decompose()`.

**Corpus baseline (1000/200):**

| Company | Raw | Clean text | Chunks |
|---|---|---|---|
| aapl | 1.5 MB | 209,393 | 292 |
| msft | 8.2 MB | 317,163 | 441 |
| tsla | 2.4 MB | 399,145 | 576 |
| **Total** | **12.1 MB** | | **1,309** |

Previous build: 768 chunks. The +70% is recovered content, not smaller chunks.

**Verification:** 60 unit + 4 integration tests pass · ruff clean · mypy strict
clean (27 files). Integration tests assert on real FY2025 figures (`416,161`,
`Apple Inc.`, `10-K`) so this regression cannot return silently.

**Docs:** ADR 0007 (inline XBRL + tables), ADR 0008 (chunking), LLD Module 1,
`interview/01-ingestion.md`.

**Carried-forward flags:**
- Character-based chunking — token-based is more correct; measure before raising size
- `ix:nonNumeric` fragments (`false`, `P1Y`) still leak short tokens
- No section awareness (Item 1A / Item 7 would be good filter metadata)
- No near-duplicate handling for repeated boilerplate

---

## What's built (Module 2 — Indexing)

`indexing/` = `embeddings` + `collection` + `indexer`, plus a thin
`scripts/index_filings.py` CLI (`--recreate`, `--dry-run`).

**Key decisions:**
- Deterministic `uuid5` point IDs → re-running the indexer overwrites instead
  of appending. Auto-generated IDs fail *silently*: the second run succeeds and
  the corpus doubles.
- Collection created explicitly with dimension verification. An existing
  collection built for a different embedding model now raises a named error
  instead of returning nonsense at query time.
- Keyword payload index on `metadata.company` — note the prefix.
  `QdrantVectorStore` nests metadata, so a filter on bare `company` matches
  nothing and raises nothing. Pinned as `COMPANY_PAYLOAD_FIELD` + a test.
- Embeddings built lazily; a missing key raises `ConfigurationError` naming the
  variable rather than a 401 from inside the OpenAI client.

**Verification:** 94 unit tests pass · ruff clean · mypy strict clean (36 files,
now covering `scripts/` too). Unit tests use in-memory Qdrant — the real local
engine — so idempotency, metadata round-trip, batching, filter scoping and the
dimension check are all genuine behaviour tests, no mocks.

**Not yet run against live infrastructure.** Docker was not running during this
session, so the OpenAI + live-Qdrant integration tests are written but
unexecuted. `--dry-run` confirms 1,309 chunks ingest cleanly.

**Docs:** ADR 0009, LLD Module 2, `interview/02-indexing.md`.

**Carried-forward flags:**
- No resume/checkpoint on a partially failed indexing run
- No orphan cleanup if a re-chunk produces fewer chunks (`--recreate` is the blunt fix)
- Sync client only — the read path needs `AsyncQdrantClient` when it enters a request handler
- Sparse vectors in Qdrant may be a better home for BM25 than an in-memory index (decide in the hybrid module)

---

## What's built (Module 3 — Dense retrieval + evaluation)

`retrieval/dense.py` + `evaluation/{dataset,harness,metrics}.py` + CLI
`scripts/evaluate_retrieval.py [--in-memory] [--top-k K...] [--audit]`.

### 📊 BASELINE — dense retrieval, 1,309 chunks, `text-embedding-3-small`

| k | Hit rate | MRR |
|---|---|---|
| 1 | 75.0% | 0.750 |
| 3 | 87.5% | 0.812 |
| 5 | 87.5% | 0.812 |
| 10 | **100.0%** | 0.833 |

Median latency ~440 ms per query (almost entirely the embedding call).
Tier 2 (natural language) = **100% at k=3**. Tier 1 (keyword) = 80%.

**Read the curve, it is the brief for Modules 4–5:** recall is *solved* by
k=10 — every answer is in the pool — while MRR moves only 0.812 → 0.833. The
gap is **precision at the top, not recall.** That is a reranker's job, and it
is now measured rather than assumed.

### The one miss, diagnosed

Tier-1 item `"derivative instruments"` expecting `"uses derivative instruments"`.

1. Verified the substring exists in the cleaned corpus — exactly once. So the
   eval item is fair and retrieval genuinely missed. (Had it not existed, the
   next hour would have been spent "fixing" retrieval for an eval bug.)
2. 11 Apple chunks mention the phrase; the target (chunk 127) sits at **rank 6**
   and does not move at k=20 or k=50.
3. Rephrased as a real question — *"Does Apple use derivative instruments to
   hedge foreign currency risk?"* — the same chunk comes back at **rank 2**.

Diagnosis: retrieval noise on a bare keyword query where 11 chunks are
near-identical in embedding space. Left unfixed on purpose — it is the case
Module 5's reranker has to beat.

**Verification:** 142 unit tests · ruff clean · mypy strict clean (45 files).

**Docs:** ADR 0010, LLD Module 3, `interview/03-retrieval-and-evaluation.md`.

**Carried-forward flags:**
- 8 eval items is thin — one item is 12.5 points. Grow to 25+ before trusting small deltas
- `"net sales"` substring is loose enough to pass by accident
- No cross-company queries ("compare Apple and Tesla") — needs multi-filter retrieval and a different metric
- Retrieval-only metrics; faithfulness needs generation + an LLM judge
- Still no live Qdrant run (Docker unavailable this session) — baseline was produced with in-memory Qdrant and the real embedding API

---

## What's built (Module 4 — Sparse + hybrid retrieval)

`retrieval/` gains `filters.py` (shared vocabulary), `sparse.py` (BM25),
`fusion.py` (RRF), `hybrid.py`. Eval CLI gains
`--strategy dense sparse hybrid` so all three are scored against **one** index
build and **one** harness.

### 📊 MEASURED — same index, same harness, same dataset

| Strategy | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| dense | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |
| sparse | 50.0% / 0.500 | 75.0% / 0.604 | 87.5% / 0.629 | 100% / 0.647 |
| hybrid | 75.0% / 0.750 | 87.5% / 0.812 | 87.5% / 0.812 | 100% / 0.833 |

**Hybrid did not beat dense.** Reported as-is. Verified it is not a bug:
`rrf_ranks` shows both retrievers contributing, and fused orderings genuinely
differ (dense `[210, 213, 209, 211, …]` vs hybrid `[213, 211, 210, 209, …]`).
The aggregate tie is a coincidence of an 8-item set.

**Why fusion could not fix the one failure — the useful part.** For
`"derivative instruments"`, the answer chunk 127 sits at rank 6 in dense and
rank 5 in sparse. Both retrievers agree it is mediocre, and RRF *rewards*
agreement. No rank-recombination scheme promotes a document every input ranked
mid-pack.

→ That is the measured case for a **cross-encoder**: something that reads query
and document together rather than recombining independent opinions. Module 5
now has a specific target, not a hope.

### Bug found by a test

RRF ties resolved by dict insertion order, so `fuse([dense, sparse])` and
`fuse([sparse, dense])` could return different rankings for identical input.
Ties are common by construction (same rank in two retrievers → identical
score). Fixed by breaking ties on the identity key. Non-determinism would have
made two eval runs incomparable — which defeats the harness's whole purpose.

**Verification:** 183 unit tests · ruff clean · mypy strict clean (52 files).

**Docs:** ADR 0011 (BM25), ADR 0012 (RRF + the negative result), LLD Module 4,
`interview/04-hybrid-retrieval.md`.

**Carried-forward flags:**
- In-memory BM25 rebuilt at process start — **first thing that breaks at scale**; fix is Qdrant native sparse vectors, deliberately deferred so adoption and relocation stay separately measurable
- `k=60` untuned; with 8 items, tuning would fit noise
- Corpus-wide IDF rather than per-company — no evidence yet it matters
- `$416,161` tokenises to `416` + `161`, so BM25 cannot match a figure as a unit
- No query expansion / multi-query — another untested lever on vocabulary mismatch

---

## What's built (Module 5 — Cross-encoder reranking)

`retrieval/rerank.py` — `RerankingRetriever` wraps **any** retriever, so the
harness can score rerank-over-dense against rerank-over-hybrid.

### 📊 MEASURED — same index, same harness, top_k=3

| Strategy | Hit rate | MRR | Median latency |
|---|---|---|---|
| dense | 87.5% | 0.812 | 464 ms |
| hybrid | 87.5% | 0.812 | 497 ms |
| **dense+rerank** | **100.0%** | **0.917** | 886 ms |
| **hybrid+rerank** | **100.0%** | **0.917** | — |

**The Module 4 prediction held.** Audited, not assumed: chunk 127 moved from
rank 6 → **rank 3**, excerpt confirms the exact expected phrase. Tier 1 went
80% → 100%.

Bonus found in the audit: *"what was Apple's total revenue this year?"* now
matches **chunk 178** — the real consolidated statement of operations
(`Products | $307,003 | $294,866 | $298,085`) — instead of chunk 201. Those
figures only exist at all because of the Module 1 inline-XBRL fix.

**`dense+rerank` == `hybrid+rerank`.** Once reranking exists, fusion adds
nothing on this dataset. BM25 is costing a second retriever per query for zero
measured gain. Kept (8 items cannot support "never"), but **the honest
recommendation today is dense + rerank**.

### 🐛 The bug worth more than the feature

The first rerank run showed **zero improvement** — because the reranker had
never run. Cohere's trial tier allows ~10 req/min, an eval issues dozens, and
`fail_open=True` did its job: logged a warning, returned un-reranked
candidates. The harness scored the fallback and reported it as the reranker's
number. The run completed; the numbers looked plausible.

**Principle: graceful degradation and honest measurement are in direct
conflict.** In a service, failing open is right. In an evaluation it is a trap.
The same component needs opposite policies, so `fail_open` is a constructor
argument and the eval CLI passes `False`.

Also fixed: provider exceptions now wrap in `RetrievalError`, so catching
`SecfilerRagError` genuinely catches everything this package raises — a raw
`cohere.TooManyRequestsError` escaping `search()` broke that contract.

**Verification:** 206 unit tests · ruff clean · mypy strict clean (54 files).

**Docs:** ADR 0013, LLD Module 5, `interview/05-reranking.md`.

**Carried-forward flags:**
- 8 eval items — 100% here is a smoke test passing, not a solved problem
- Hybrid's value is unproven post-reranking and still in the code
- No caching of query embeddings or rerank calls; both are deterministic
- Rate limits make a full eval sweep take minutes of backoff
- Still retrieval-only — nothing measures whether the *answer* is faithful

---

## What's next

**Module 6 — Answer generation (the "G" in RAG)**

Retrieval is done: 3 precise chunks, 100% hit rate. Now turn them into a cited
answer.

1. `generation/prompts.py` — grounding instruction, numbered context blocks so
   the model can cite them
2. `generation/chain.py` — LCEL: retriever → context assembly → LLM → parsed
   answer + citations
3. **Refusal path is mandatory.** An empty or weak retrieval must never reach
   the model with an instruction to answer anyway — "not in these filings" is a
   correct answer and the main guardrail against invention
4. Citations tied to `(company, chunk_id)` so every claim is traceable
5. Token budgeting — count, do not hope
6. Print the fully rendered prompt at least once; what you think you sent and
   what the template produced differ more often than anyone admits
7. Faithfulness evaluation (LLM-as-judge) — the harness currently measures
   retrieval only, and generation needs its own metric
8. The mangled-table problem finally becomes visible here: retrieval can hide a
   missing figure, generation cannot

**Also outstanding:** run against live Qdrant once Docker is available —
`docker compose up -d && uv run python scripts/index_filings.py` then
`uv run pytest -m integration`.

---

## Working rules

1. Production quality — no demo shortcuts.
2. Incremental modules — ship one stage at a time with tests + docs.
3. Every major decision gets an ADR + interview notes.
4. Explain *why*, not only *what* — LangChain abstraction vs core RAG concept.
5. Eval numbers before claims.
6. Never keep a line you cannot defend in an interview.

---

*Last updated: Module 5 complete — reranking took hit rate 87.5% → 100% and MRR 0.812 → 0.917; the Module 4 prediction held and was audited. Found that fail-open silently corrupted the first measurement. Retrieval is done. Next: Module 6 (Answer generation).*
