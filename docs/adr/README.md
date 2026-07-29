# Architecture Decision Records

An ADR captures **one decision**, the context that forced it, the alternatives
that were rejected, and the consequences accepted. It is written when the
decision is made, while the reasoning is still fresh, and it is never edited
afterwards — a decision that changes gets a *new* ADR that supersedes the old
one.

## Why bother

- Six months later, "why is fusion rank-based instead of score-based?" has an
  answer that is not a guess.
- In an interview, "we chose X" is weak; "we chose X over Y because Z, and
  accepted trade-off W" is the answer that lands.
- It stops decisions being silently relitigated every time someone new reads
  the code.

## Format

```
# ADR NNNN — Title
Status | Date
## Context      — the forces at play
## Decision     — what we do
## Alternatives — what we rejected, and why
## Consequences — what this costs us
## Interview angle — the question this invites
```

## Index

| # | Decision | Status |
|---|---|---|
| [0001](0001-clean-slate-langchain-rebuild.md) | Rebuild from a clean slate on LangChain | Accepted |
| [0002](0002-src-layout-and-stage-packages.md) | `src/` layout with one-way stage packages | Accepted |
| [0003](0003-environment-based-configuration.md) | Typed environment configuration via pydantic-settings | Accepted |
| [0004](0004-stdlib-structured-logging.md) | Standard-library logging with a JSON formatter | Accepted |
| [0005](0005-incremental-dependencies.md) | Add dependencies only when a module needs them | Accepted |
| [0006](0006-raw-filings-in-version-control.md) | Track raw filings in git | Accepted |
| [0007](0007-preserve-inline-xbrl-values.md) | Preserve inline-XBRL values; flatten tables into rows | Accepted |
| [0008](0008-recursive-chunking-strategy.md) | Recursive character chunking at 1000/200 | Accepted |
