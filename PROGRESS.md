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
| 2 | Indexing — OpenAI embeddings + Qdrant upsert | 🔜 **NEXT** |
| 3 | Dense retrieval + company filter + eval harness | Not started |
| 4 | Sparse (BM25) + hybrid RRF fusion | Not started |
| 5 | Cross-encoder reranking | Not started |
| 6 | Answer generation (cited answers) | Not started |
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

## What's next

**Module 2 — Indexing**

1. Add `langchain-openai` + `langchain-qdrant` + `qdrant-client` (deps land with the module)
2. Add Qdrant + embedding settings (`QDRANT_URL`, collection name, model, batch size)
3. Embed chunks with `text-embedding-3-small` (1536-d), batched
4. Collection lifecycle: create/verify, cosine distance, dimension check
5. Deterministic point IDs — `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")` — so re-indexing overwrites instead of duplicating
6. Payload index on `company` so filtering stays cheap
7. Integration test against live Qdrant; unit tests with a fake embedder
8. Docs: ADR for point-ID scheme + single-collection choice, LLD, interview Q&A

---

## Working rules

1. Production quality — no demo shortcuts.
2. Incremental modules — ship one stage at a time with tests + docs.
3. Every major decision gets an ADR + interview notes.
4. Explain *why*, not only *what* — LangChain abstraction vs core RAG concept.
5. Eval numbers before claims.
6. Never keep a line you cannot defend in an interview.

---

*Last updated: Module 1 complete — ingestion built, inline-XBRL data-loss bug found and fixed, 1,309 chunks baselined. Next: Module 2 (Indexing).*
