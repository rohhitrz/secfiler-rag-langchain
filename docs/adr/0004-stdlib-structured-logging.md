# ADR 0004 — Standard-library logging with a JSON formatter

**Status:** Accepted · **Date:** 2026-07-29

## Context

The previous build had no logging — diagnostics were `print()` calls inside
`__main__` blocks. That is unusable in a service: no levels, no timestamps, no
context, and nothing an aggregator can query.

A RAG pipeline particularly needs *structured* logs. "Which company, how many
chunks, which retriever, how many candidates, how long" are all fields you want
to filter and aggregate on — not prose to grep.

## Decision

Use the **standard library `logging`** module, configured once in
`core/logging.py`:

- `configure_logging(level, fmt, force)` — idempotent, installs exactly one
  handler on the root logger
- Two formatters: `console` (human-readable, local dev) and `json` (one object
  per line, for aggregators)
- Structured context via the standard `extra=` argument, promoted to top-level
  JSON fields
- Project logger tree (`secfiler_rag.*`) at the configured level; **third-party
  loggers pinned at WARNING** so `LOG_LEVEL=DEBUG` does not drown our output in
  `httpx` chatter
- `get_logger(__name__)` in every module

## Alternatives

**`structlog`.** Genuinely good, and the usual answer for structured logging.
Rejected here because every dependency we use — LangChain, httpx,
qdrant-client, openai — logs through the stdlib. Configuring the stdlib root
logger captures *their* diagnostics too. With a separate logging library you
own your lines and lose theirs, which is exactly the output you need when a
retrieval call hangs. (`structlog` can bridge to stdlib, but that is a second
dependency plus glue for a problem 60 lines of `Formatter` already solves.)

**`loguru`.** Rejected: pleasant API, but it replaces the stdlib idiom
wholesale and makes library log interception awkward.

**`print()`.** Rejected: no levels, no structure, unconditionally to stdout.

## Consequences

- We maintain ~60 lines of `JsonFormatter` — small, and fully understood.
- Callers must remember `extra={...}` rather than f-strings. This is a review
  convention, documented in the development guide.
- `configure_logging` must be called explicitly at each entry point (script,
  test, future API startup). Deliberate: libraries must not configure logging
  on import, because that steals the decision from the application embedding
  them.
- Idempotency is tested — repeated calls must not stack handlers, the classic
  cause of every line printing three times.

## Interview angle

> **Q: Why not structlog?**
>
> Because I care more about capturing LangChain's and httpx's logs than about
> a nicer API for my own. Everything in the dependency tree logs through the
> stdlib, so configuring the stdlib root logger means one handler sees
> everything. A JSON formatter is about sixty lines, and I would rather own
> those than add a dependency and still need a bridge to the stdlib.
>
> **Follow-up: how do you get structured fields without structlog?**
>
> The `extra=` argument. `log.info("indexed filing", extra={"company": "aapl",
> "chunks": 199})` — my formatter promotes anything that is not a reserved
> `LogRecord` attribute to a top-level JSON field.
>
> **Follow-up: why is third-party logging pinned to WARNING?**
>
> Because `LOG_LEVEL=DEBUG` is something you set when you are already in
> trouble, and if it turns on `httpx` request debug you lose your own output in
> the noise. Our tree gets the configured level; everyone else stays at
> WARNING unless I deliberately turn them up.
