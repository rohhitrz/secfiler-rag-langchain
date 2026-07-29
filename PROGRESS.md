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
| 1 | Ingestion — HTML load, clean, chunk → `Document` | 🔜 **NEXT** |
| 2 | Indexing — OpenAI embeddings + Qdrant upsert | Not started |
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

## What's next

**Module 1 — Ingestion**

1. Load HTML from `data/raw/`
2. Strip scripts/styles/inline-XBRL with BeautifulSoup
3. Chunk with overlap; attach metadata (`company`, `chunk_id`, `source`)
4. Emit `langchain_core.documents.Document`
5. Unit tests on tiny HTML fixtures (not full 8MB MSFT in unit tests)
6. Docs: LLD for ingestion, failure modes (tables / mangled `$`), interview Q&A

---

## Working rules

1. Production quality — no demo shortcuts.
2. Incremental modules — ship one stage at a time with tests + docs.
3. Every major decision gets an ADR + interview notes.
4. Explain *why*, not only *what* — LangChain abstraction vs core RAG concept.
5. Eval numbers before claims.
6. Never keep a line you cannot defend in an interview.

---

*Last updated: Module 0 complete — repo cleaned, structure locked, foundational docs written. Next: Module 1 (Ingestion).*
