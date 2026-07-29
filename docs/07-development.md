# Development Guide

## 1. The loop

```bash
uv run pytest -m "not integration"     # fast: no network, no Docker
uv run ruff format src tests           # format
uv run ruff check src tests --fix      # lint
uv run mypy                            # types
```

Run all four before every commit. They take seconds; a broken `main` costs
more.

## 2. How a module gets added

Each module follows the same sequence. It is deliberately documentation-first
because the design decisions are the part worth defending later.

1. **Write the contract** — add the module's section to `docs/02-lld.md`:
   public functions, input/output types, metadata keys, failure modes.
2. **Write an ADR** if the module makes a non-obvious choice (a library, a
   strategy, a trade-off you might be asked to justify).
3. **Write the tests** from the contract, before the implementation. Tests
   written afterwards tend to describe the code you wrote rather than the
   behaviour you wanted.
4. **Implement** the smallest thing that satisfies the contract.
5. **Add dependencies only now**, if the implementation genuinely needs them.
6. **Measure** if the module touches retrieval or generation quality — a number
   from the eval harness, not an impression.
7. **Update** `PROGRESS.md` and `docs/interview/` with the questions this
   module invites.

## 3. Coding standards

**Type hints are mandatory.** `mypy --strict` runs over `src/` and `tests/`.
Untyped code is unreviewable at a distance and hides whole classes of bug.

**Docstrings on every public module, class and function** (`ruff` rule `D`,
Google convention). Say *why*, not *what* — the signature already says what.

```python
# Not useful: restates the signature
def chunk_text(text: str, size: int) -> list[str]:
    """Chunk text into pieces of a given size."""

# Useful: records the decision
def chunk_text(text: str, size: int) -> list[str]:
    """Split cleaned filing text into overlapping windows.

    Overlap exists because a fact split across a boundary is unretrievable by
    either chunk. 200 chars is ~20% of the window — enough to survive a
    sentence break without materially inflating index size.
    """
```

**No `os.environ` outside `config/settings.py`.** Every knob goes through
`get_settings()`. This is enforced by review, and it is what makes the system
configurable without a code search.

**No module-level side effects.** No API client constructed at import time, no
file read at import time. Import must be free. The previous build instantiated
`OpenAI()` and `QdrantClient()` at module scope, which meant importing a module
to run one unit test required live credentials.

**Log with structure:**

```python
log = get_logger(__name__)
log.info("indexed filing", extra={"company": "aapl", "chunks": 199})
```

Not `log.info(f"indexed aapl with 199 chunks")` — the f-string is unqueryable
once it reaches a log aggregator.

**Raise typed errors** from `core.exceptions`, never bare `Exception`.

## 4. Testing strategy

| Kind | Marker | Runs | Uses |
|---|---|---|---|
| Unit | none | Every save | Tiny in-repo fixtures, no network |
| Integration | `@pytest.mark.integration` | Pre-commit / CI | Real Qdrant, real APIs |

```bash
uv run pytest -m "not integration"   # the default loop
uv run pytest -m integration         # needs Docker + keys
```

**Do not put the 8 MB Microsoft filing in a unit test.** Unit tests use small
handwritten HTML fixtures that exercise the *shape* of the problem (a nested
inline-XBRL tag, a table, a `<script>` block). The full filings belong in
integration tests and eval runs.

**Test behaviour, not implementation.** `test_settings.py` asserts that
secrets do not leak into `repr()` — a property that must hold however the class
is written. It does not assert which pydantic version produced it.

## 5. Dependency policy

A package is added when the module that needs it lands, in the same commit.
See [ADR 0005](adr/0005-incremental-dependencies.md).

```bash
uv add langchain-qdrant          # runtime
uv add --group dev pytest-cov    # tooling
```

Both update `pyproject.toml` *and* `uv.lock`. Commit the lockfile.

## 6. Git conventions

Conventional commits, imperative mood:

```
feat(ingestion): add inline-XBRL stripping
fix(retrieval): use (company, chunk_id) as RRF identity key
docs(adr): record RRF over weighted score fusion
test(config): cover secret redaction in repr
chore(deps): add langchain-qdrant
```

One logical change per commit. A commit that adds a feature, reformats three
files and bumps a dependency cannot be reviewed or reverted cleanly.

## 7. Evaluation discipline

- Never claim an improvement without a number from the harness.
- Change one variable at a time. Chunk size *and* reranker together tells you
  nothing about either.
- Record the number in `PROGRESS.md` with the config that produced it.
- When a metric jumps, **audit the passing items** before celebrating. A hit
  can be a substring accident rather than real retrieval — this caught two
  false positives in the previous build.
