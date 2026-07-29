# ADR 0003 — Typed environment configuration via pydantic-settings

**Status:** Accepted · **Date:** 2026-07-29

## Context

The previous build had a nine-line `config.py` that instantiated
`settings = Settings()` at module import. Three consequences followed:

1. Importing *any* module that touched config required a valid `OPENAI_API_KEY`
   — so unit tests needed live credentials.
2. Values that were not in `Settings` (collection name `"filings"`, `top_k`,
   chunk size, RRF `k`) were hardcoded across several files, so changing one
   meant grepping.
3. Nothing validated that a `.env` value was well-formed until it was used.

## Decision

One `Settings` class (pydantic-settings `BaseSettings`), accessed **only**
through a cached factory:

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings: ...
```

Rules:

- **No `os.environ` anywhere outside `config/settings.py`.**
- `frozen=True` — settings are immutable after construction; a value that
  changes under you mid-run is unreproducible.
- `extra="ignore"` — the `.env` is shared with other tools; unknown keys must
  not crash the app.
- Secrets typed as `SecretStr` so they cannot leak into a `repr()`, a
  traceback, or a log line by accident.
- **No prefix** on env names: `OPENAI_API_KEY` and `LANGSMITH_*` are read
  directly from the process environment by their own SDKs. A custom prefix
  would mean maintaining two names for one secret.
- Fields are added when the module that needs them lands, so the class stays a
  truthful inventory of what the system uses.

## Alternatives

**Module-level `settings = Settings()`.** Rejected — that is precisely the
import-time side effect being removed. `lru_cache` gives the same singleton
ergonomics lazily.

**Plain `os.getenv` with defaults.** Rejected: no types, no validation, no
single place to see every knob, and typos become silent defaults.

**A YAML/TOML config file.** Rejected: secrets do not belong in files that are
easy to commit, and 12-factor environment variables are what every deployment
target already speaks. A file layer can be added later underneath the same
`Settings` interface if config grows structurally complex.

**Dependency-inject settings into every function.** Partially adopted — call
sites that need explicit settings can take a `Settings` parameter; the cached
factory is the default for convenience. Threading settings through every
signature is purity at a real ergonomic cost.

## Consequences

- Tests must clear the cache (`get_settings.cache_clear()`) — handled by an
  autouse fixture — and must avoid reading the developer's real `.env`
  (handled by an `IsolatedSettings` subclass with `env_file=None`).
- Misconfiguration fails at startup with a field-level error naming the
  offending variable, rather than as an `AttributeError` deep in a retriever.
- `.env.example` must list every variable, or the fail-fast guarantee is
  useless to a newcomer.

## Interview angle

> **Q: Why not just read environment variables where you need them?**
>
> Three reasons. Typos become silent defaults instead of startup errors — the
> worst kind of production bug. There is no single place to see what is
> configurable. And it is untestable: you cannot override a value without
> mutating global state. A typed settings object validates once, at the
> boundary, and everything downstream receives values it can trust.
>
> **Follow-up: why `lru_cache` instead of a module-level instance?**
>
> Import-time side effects. My previous version built the settings object and
> API clients at import, so running a single unit test required a real OpenAI
> key. Caching the *factory* gives the same singleton behaviour but defers the
> work to first use — and makes it clearable in tests.
>
> **Follow-up: how do you keep secrets out of logs?**
>
> `SecretStr`. It renders as `**********` in `repr()` and tracebacks; you have
> to call `.get_secret_value()` deliberately. That turns "do not log secrets"
> from a code-review convention into a type-level property.
