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
└── IngestionError
```

More leaf types are added when new failure modes appear (e.g. `IndexingError`, `RetrievalError`).

---

## Module 1 — Ingestion (planned)

### Public surface (intended)

```text
load_filing(path: Path, company: str) -> str
chunk_documents(text: str, *, company: str, source: str, ...) -> list[Document]
ingest_company(company: str) -> list[Document]
ingest_all() -> list[Document]
```

### Metadata contract on every `Document`

| Key | Type | Meaning |
|---|---|---|
| `company` | `str` | `aapl` / `msft` / `tsla` |
| `chunk_id` | `int` | Position in that company's chunk list |
| `source` | `str` | Filename or relative path |

### Chunking defaults (baseline)

- Fixed size + overlap (exact numbers locked when Module 1 lands)
- Structure-aware / semantic chunking deferred until measured need

### Failure modes

- Missing file → `IngestionError`
- Empty cleaned text → `IngestionError`
- Unknown company key → `IngestionError` or validation error

---

## Module 2 — Indexing (planned)

### Public surface (intended)

```text
ensure_collection(client, ...) -> None
index_documents(docs: list[Document]) -> int  # points upserted
```

### Collection

- Name: `filings` (from settings once added)
- Distance: Cosine
- Vector size: embedding model dimension (1536 for `text-embedding-3-small`)

### Idempotency

Same `(company, chunk_id)` → same point ID → re-index overwrites, does not duplicate.

---

## Module 3+ — Retrieval / evaluation / generation

Contracts will be filled when those modules land. Standing rules:

- Retrievers implement LangChain `BaseRetriever` (or a thin adapter)
- Eval harness accepts a retriever + dataset; returns metrics
- Generation always receives an explicit list of context docs (no hidden global state)

---

## Dependency rule (enforced by design)

```text
ingestion  →  indexing  →  retrieval  →  generation
                  ↑              ↑
            evaluation     observability
config / core ← used by all; import nothing from stages
```

`core` must never import `ingestion` / `indexing` / `retrieval` / `generation`.
