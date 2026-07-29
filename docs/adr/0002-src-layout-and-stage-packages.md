# ADR 0002 — `src/` layout with one-way stage packages

**Status:** Accepted · **Date:** 2026-07-29

## Context

The previous build used `src/secfiler_rag/` with a single `rag/` package
holding `ingest.py`, `embed.py`, `index.py`, `retrieve.py`, `search.py`,
`fuse.py`, `rerank.py`, `evaluate.py`. Every file could import every other, and
`fuse.py` had grown into four responsibilities. Nothing prevented a cycle.

At eight files this is survivable. At twenty-five, with an API layer and
agentic routing, it is not — and the cost is not aesthetic: cyclic packages
cannot be tested, deployed or reasoned about independently.

## Decision

Keep the `src/` layout, and split the package by **pipeline stage**, with a
strictly one-way dependency rule:

```
config ─┐
core   ─┤
        ▼
  ingestion → indexing → retrieval → generation
                            ▲             ▲
                      evaluation    observability
```

- Stages import leftward and from `config`/`core` only.
- `core` imports nothing from any stage.
- Nothing imports `evaluation` — measurement cannot influence behaviour.

`src/` (rather than a top-level package) means the repo root is not importable,
so tests exercise the *installed* package and packaging errors surface locally.

## Alternatives

**Flat layout** (`secfiler_rag/` at the root). Rejected: imports silently
resolve to the working directory, so a module missing from the wheel passes
every local test and fails on install.

**Layer by type** (`models/`, `services/`, `utils/`). Rejected: it scatters one
feature across three directories, and `utils/` becomes the place where
everything that does not fit goes to rot. Changing the chunking strategy should
touch one package, not three.

**Keep one `rag/` package, split files only.** Rejected: file boundaries are
not enforced boundaries. Package boundaries with a stated import rule are
reviewable — "does this import point rightward?" is a yes/no question.

## Consequences

- More directories than the current amount of code strictly needs. Accepted:
  the structure is the design statement, and each `__init__.py` documents that
  package's single responsibility.
- The import rule is enforced by review, not tooling. If it is ever violated in
  practice, add an `import-linter` contract to CI.
- Placing a new file requires deciding which stage owns it — which is the
  design work, surfaced early instead of discovered late.

## Interview angle

> **Q: Why not just organise by file?**
>
> Because file boundaries do not stop cyclic dependencies and package
> boundaries do. My previous version had one `rag/` package where `fuse.py`
> imported the eval harness, the vector search, the reranker and the eval
> dataset — so I could not test fusion without a live Qdrant. Splitting by
> pipeline stage with a one-way import rule means retrieval can be tested
> without the write path, and a read-only service never needs embedding
> credentials.
>
> **Follow-up: how do you enforce it?**
>
> Review today, because the codebase is small and I am the only author. The
> moment there is a second contributor I would add an `import-linter` contract
> to CI — the rule is only real if something checks it.
