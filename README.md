# secfiler-rag

Production-grade **Advanced RAG** over SEC 10-K filings (Apple, Microsoft, Tesla).

Built with **LangChain**, **Qdrant**, **OpenAI embeddings**, and **LangSmith** — structured as a real software system, not a notebook demo.

| | |
|---|---|
| **Corpus** | FY2025 10-K HTML for AAPL, MSFT, TSLA |
| **Python** | 3.12+ · packaged with `uv` |
| **Status** | Dense retrieval measured — 87.5% hit rate @ k=5, 100% @ k=10 |

---

## Why this project exists

Most RAG demos stop at “embed → similarity search → LLM.” This repo is built to demonstrate the engineering around that path:

- Clear module boundaries and a one-way pipeline
- Environment-based, typed configuration
- Structured logging and a deliberate exception hierarchy
- Retriever-agnostic evaluation (honest A/B)
- Architecture Decision Records and interview-ready explanations

If it is on a resume, every major choice should be defensible in a senior AI engineering interview.

---

## Architecture (one screen)

```
data/raw/*.htm
       │
       ▼
┌─────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│  ingestion  │───▶│  indexing  │───▶│  retrieval  │───▶│ generation │
│ HTML→chunks │    │ embed+Qdrant│    │ dense/hybrid│    │ cited answer│
└─────────────┘    └────────────┘    └─────────────┘    └────────────┘
                           │                 ▲
                           │         evaluation · observability
                           ▼
                        Qdrant
```

Cross-cutting: `config` (settings), `core` (logging, errors), `evaluation`, `observability` (LangSmith).

Full design: [`docs/01-hld.md`](docs/01-hld.md) · folder rationale: [`docs/03-folder-structure.md`](docs/03-folder-structure.md).

---

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Qdrant)

### Install

```bash
git clone <this-repo>
cd secfiler-rag-langchain
uv sync --group dev
cp .env.example .env
# add OPENAI_API_KEY when you reach indexing / generation
```

### Run Qdrant

```bash
docker compose up -d
# dashboard: http://localhost:6333/dashboard
```

### Build the index

```bash
uv run python scripts/index_filings.py --dry-run
```

```bash
uv run python scripts/index_filings.py
```

### Measure retrieval

```bash
uv run python scripts/evaluate_retrieval.py --in-memory --top-k 1 3 5 10 --audit
```

### Tests

```bash
uv run pytest -m "not integration"
```

### Lint / type-check

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

---

## Repository map

```
data/raw/                  # tracked SEC filings (source of truth)
docs/                      # HLD, LLD, ADRs, guides, interview notes
evals/datasets/            # versioned eval fixtures (seed set carried over)
scripts/                   # thin CLIs (later)
src/secfiler_rag/
  config/                  # pydantic-settings
  core/                    # logging, exceptions
  ingestion/               # HTML → Documents          ✅
  indexing/                # embed + Qdrant            ✅
  retrieval/               # search strategies         ✅ dense
  generation/              # cited answers             (Module 6)
  evaluation/              # harness + metrics         ✅
  observability/           # LangSmith                 (Module 7)
tests/unit|integration/
```

---

## Documentation

| Doc | Purpose |
|---|---|
| [Docs index](docs/README.md) | Start here — reading order |
| [High-Level Design](docs/01-hld.md) | System shape, components, trade-offs |
| [Low-Level Design](docs/02-lld.md) | Module contracts (grows with code) |
| [Folder structure](docs/03-folder-structure.md) | Why each package exists |
| [Data flow](docs/04-data-flow.md) | End-to-end pipeline |
| [Request lifecycle](docs/05-request-lifecycle.md) | Query path (once serving lands) |
| [Setup](docs/06-setup.md) | Environment and tooling |
| [Development](docs/07-development.md) | How we add features |
| [Debugging](docs/08-debugging.md) | How to diagnose failures |
| [Failure modes](docs/09-failure-modes.md) | Classic RAG failure taxonomy |
| [Scaling & performance](docs/10-scaling-and-performance.md) | What changes under load |
| [ADRs](docs/adr/) | Architecture decisions |
| [Interview prep](docs/interview/) | Q&A per component |
| [PROGRESS.md](PROGRESS.md) | Session state / syllabus |

---

## Design principles

1. **One responsibility per package** — pipeline stages do not import “forward.”
2. **Core RAG concepts over framework magic** — LangChain is orchestration; chunking, fusion, filters, and eval design are ours.
3. **Fail fast at the boundary** — bad config and bad inputs raise typed errors early.
4. **Measure before claiming** — eval harness stays strategy-agnostic.
5. **Dependencies earn their place** — a package is added when the module that needs it lands.

---

## Roadmap (short)

1. ✅ Module 0 — clean repo, structure, foundation docs
2. ✅ Module 1 — ingestion (HTML → cleaned text → 1,309 chunks)
3. ✅ Module 2 — indexing (OpenAI embeddings + Qdrant, idempotent re-index)
4. ✅ Module 3 — dense retrieval + eval harness (first measured baseline)
5. 🔜 Module 4 — hybrid search (BM25 + RRF)
6. Rerank → generation → serving

Details in [`PROGRESS.md`](PROGRESS.md).

---

## License

Private / portfolio use unless otherwise stated.
