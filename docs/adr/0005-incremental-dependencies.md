# ADR 0005 — Add dependencies only when a module needs them

**Status:** Accepted · **Date:** 2026-07-29

## Context

The previous `pyproject.toml` declared eight runtime dependencies —
`fastapi`, `uvicorn[standard]`, `cohere`, `openai`, `qdrant-client`,
`rank-bm25`, `beautifulsoup4`, `pydantic-settings` — for a codebase whose only
entry point was a health-check endpoint and a handful of `__main__` blocks.
`fastapi` and `uvicorn` were installed for months before anything served a
request.

The cost is not disk space. It is that the dependency list stops describing the
system: a reader cannot tell what is actually used, and neither can a security
scanner triaging a CVE.

## Decision

A package enters `pyproject.toml` **in the same commit as the module that
imports it**.

Current runtime set (foundation only):

| Package | Used by |
|---|---|
| `langchain-core` | `Document` type — the contract between every stage |
| `langchain-text-splitters` | Chunking (ingestion, next module) |
| `langsmith` | Tracing (observability) |
| `beautifulsoup4` | HTML cleaning (ingestion) |
| `pydantic` / `pydantic-settings` | Typed configuration |

Deliberately **not** installed yet: `qdrant-client`, `langchain-openai`,
`langchain-qdrant`, `rank-bm25`, `cohere`, `fastapi`, `uvicorn`. Each arrives
with its module.

Dev tooling is exempt — `pytest`, `ruff`, `mypy` are needed from commit one.

## Alternatives

**Install the full stack up front.** Rejected: the dependency list becomes
aspirational rather than descriptive, and unused packages still carry CVEs,
resolver constraints and install time.

**Optional extras** (`pip install .[serving]`). A good pattern for a library
with genuinely optional features. Rejected here as premature — this is one
application with one deployment shape.

## Consequences

- `uv add` runs several more times over the project's life. Trivial.
- The dependency graph is always a true statement about the code, which makes
  "why is this here?" answerable at any commit.
- Version resolution happens incrementally, so a conflict is attributable to
  the package that just arrived rather than to a wall of eight.
- Docs must state which module brings which dependency — done in the table
  above and in [`06-setup.md`](../06-setup.md).

## Interview angle

> **Q: Isn't this just churn? Why not install what you know you will need?**
>
> Because a dependency list that includes things nothing imports stops being
> information. In my previous version `fastapi` and `uvicorn` were installed
> for months before a single endpoint existed — so if a CVE had landed on
> either, triage would have started with "do we even use this?" Adding a
> package with the module that imports it keeps the answer to that question
> obvious, and it makes resolution conflicts attributable to one change.
>
> **Follow-up: how do you avoid a big-bang resolution problem later?**
>
> The lockfile. `uv.lock` is committed, so every add is an explicit, reviewable
> diff and every machine resolves identically.
