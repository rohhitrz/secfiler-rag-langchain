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
