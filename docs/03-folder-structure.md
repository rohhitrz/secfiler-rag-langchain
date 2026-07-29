# Folder Structure — and why each directory exists

A directory layout is an architecture claim. This one claims: *the pipeline has
four stages, two cross-cutting concerns, and no cycles.* If a file is hard to
place, the design is drifting — that is the point of a strict layout.

---

## 1. Top level

```
secfiler-rag-langchain/
├── data/raw/               # the three SEC 10-K filings — source of truth
├── docs/                   # HLD, LLD, ADRs, guides, interview notes
├── evals/datasets/         # versioned eval fixtures (data, not code)
├── scripts/                # thin entry points / one-off operational scripts
├── src/secfiler_rag/       # the package (importable, installable)
├── tests/                  # unit + integration
├── docker-compose.yml      # Qdrant for local development
├── pyproject.toml          # deps + ruff + mypy + pytest config, one file
├── PROGRESS.md             # session state, syllabus, locked decisions
└── README.md
```

### Why `src/` layout instead of a top-level package?

With a flat layout (`secfiler_rag/` at the repo root), `import secfiler_rag`
resolves to the *working directory* — so your tests import the source tree, not
the installed package. Anything broken in packaging (a missing `__init__.py`, a
module left out of the wheel) stays invisible until someone installs it.

With `src/`, the repo root is not importable. Tests can only import what was
actually installed (`uv sync` installs the project in editable mode), so
packaging bugs surface locally instead of in production.

**Cost:** you must install the project before running tests. `uv run pytest`
does that for you, so the cost is ~zero and the guarantee is real.

### Why `evals/` outside `src/`?

`evals/datasets/` holds *data* — question/answer fixtures. It is version
controlled because eval data is measurement infrastructure (a changed eval set
invalidates every prior number), but it is not application code and should not
ship inside the wheel. The *harness* that consumes it lives in
`src/secfiler_rag/evaluation/`.

That split — **harness in the package, data outside** — is what lets you swap
datasets without touching code and diff dataset changes in review.

---

## 2. Inside the package

```
src/secfiler_rag/
├── config/           # typed settings from the environment
├── core/             # logging, exception hierarchy — no RAG semantics
├── ingestion/        # Stage 1: HTML → cleaned text → Document chunks
├── indexing/         # Stage 2: Document → embeddings → Qdrant points
├── retrieval/        # Stage 3: query → ranked Documents
├── generation/       # Stage 4: Documents + query → cited answer
├── evaluation/       # cross-cutting: harness + metrics
└── observability/    # cross-cutting: LangSmith tracing
```

### The dependency rule

```
config ──┐
core   ──┤
         ▼
   ingestion ──▶ indexing ──▶ retrieval ──▶ generation
                                  ▲              ▲
                            evaluation     observability
```

Arrows are the **only** legal import directions.

- Stages import *leftward* and from `config`/`core`. Never rightward.
- `core` imports nothing from stages. If `core` ever needs to know what a
  `Document` is, the abstraction has leaked.
- `evaluation` imports retrievers/chains to *measure* them; nothing imports
  `evaluation`. Measurement must never be able to change behaviour.

**Why this matters beyond tidiness:** a cycle between `retrieval` and
`indexing` means you cannot test retrieval without the write path, cannot
deploy a read-only service without embedding credentials, and cannot reason
about what a change breaks. One-way imports keep those three properties.

### Why these boundaries, specifically?

| Package | Owns | Changes when… | Must NOT know |
|---|---|---|---|
| `ingestion` | HTML quirks, cleaning, chunking | The corpus format or chunk strategy changes | That Qdrant or embeddings exist |
| `indexing` | Embedding calls, collection lifecycle, upserts | You switch embedding model or vector store | How HTML was cleaned |
| `retrieval` | Dense / sparse / hybrid / rerank, filters | You try a new search strategy | How points got written |
| `generation` | Prompts, context assembly, citations | Prompt or model changes | How context was found |
| `evaluation` | Metrics, harness | You add a metric | Which strategy is under test |
| `observability` | LangSmith wiring | Tracing backend changes | Business logic |

The middle column is the real test: **each package has exactly one reason to
change.** That is the Single Responsibility Principle applied at package level,
and it is what makes "swap the reranker" a one-package diff.

### Why `config` and `core` are separate

`config` answers *"what are the settings?"*. `core` answers *"what vocabulary
does everything share?"* (loggers, error types). They are separate because
`core` is stable and `config` grows every module. Merging them means every new
setting touches the module that everything imports.

---

## 3. Tests mirror the package

```
tests/
├── conftest.py          # shared fixtures (env isolation)
├── unit/                # no network, no Docker, milliseconds
└── integration/         # real Qdrant / real API — marked, deselectable
```

Unit tests run on every save (`uv run pytest -m "not integration"`). Integration
tests run before commit and in CI where services exist. Mixing the two produces
a suite nobody runs.

`tests/`, `tests/unit/` and `tests/integration/` each have an `__init__.py` so
module names stay unique — without them, `tests/unit/test_retrieval.py` and
`tests/integration/test_retrieval.py` collide under pytest's importer.

---

## 4. Where does a new file go?

| If it… | It belongs in… |
|---|---|
| Parses or splits source documents | `ingestion/` |
| Talks to the embedding API or writes to Qdrant | `indexing/` |
| Turns a query into ranked documents | `retrieval/` |
| Builds a prompt or calls the chat model | `generation/` |
| Computes a score over a fixed dataset | `evaluation/` |
| Reads an environment variable | `config/settings.py` — nowhere else |
| Is a shared error or logger concern | `core/` |
| Is a runnable command a human types | `scripts/` (thin: parse args, call the package) |

**`scripts/` must stay thin.** Logic in a script is logic you cannot unit test
or import. A script parses arguments, calls one package function, and prints.
