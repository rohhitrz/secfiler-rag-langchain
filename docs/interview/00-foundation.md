# Interview — Module 0: Foundation

Covers repository structure, configuration, logging, testing setup, and
dependency policy. These are the questions that separate "built a RAG demo"
from "built a system."

---

## Q1. Walk me through your project structure.

**Answer.** It is a `src/` layout with the package split by pipeline stage:
`ingestion → indexing → retrieval → generation`, plus two cross-cutting
packages, `evaluation` and `observability`, and two foundation packages,
`config` and `core`.

The rule that makes it a design rather than a folder tree is that imports go
one way only. Stages import leftward and from `config`/`core`; `core` imports
nothing from any stage; nothing imports `evaluation`.

Three concrete properties fall out of that:

- Retrieval can be tested without the write path
- A read-only service does not need embedding credentials
- Changing the chunking strategy touches exactly one package

**Follow-up: why `src/` instead of a package at the repo root?**

With a flat layout, `import secfiler_rag` resolves to the working directory, so
tests import the source tree rather than the installed package — a module
missing from the wheel passes every local test. With `src/`, the root is not
importable, so tests run against what was actually installed.

**Follow-up: why not organise by type — models, services, utils?**

Because that scatters one feature across three directories, and `utils/`
becomes where everything that does not fit goes to rot. Organising by pipeline
stage means each package has exactly one reason to change.

---

## Q2. Why did you rewrite a working system?

**Answer.** The first version proved the retrieval mechanics — BM25 baseline,
vector search, RRF fusion, cross-encoder reranking, an eval harness at 8/8. I
carried every one of those *decisions* forward. What I did not carry forward
was the structure: one file with four responsibilities, API clients constructed
at import time so tests needed live credentials, pipelines driven by `__main__`
blocks, and no tests.

The rewrite is about the engineering around the RAG. The fact that the
retrieval decisions transferred unchanged is evidence they were sound.

**Follow-up: doesn't LangChain hide what you learned?**

It would have if I had started there. I implemented BM25, RRF and reranking by
hand first, so I know what `EnsembleRetriever` does internally — including that
its default weighting is not RRF — and where its defaults would hurt me.
Mechanics first, framework second, deliberately.

**Follow-up: what does LangChain actually buy you here?**

Three things: a common `Document` type so stages have a stable contract; the
`BaseRetriever` interface so the eval harness can score any strategy without
knowing which; and LangSmith tracing essentially for free. The RAG design —
chunk strategy, fusion, filters, candidate widths, eval methodology — is still
mine. LangChain does orchestration and I/O.

---

## Q3. How is configuration handled?

**Answer.** One `Settings` class built on pydantic-settings, reached only
through a cached `get_settings()` factory. No `os.environ` anywhere else in the
codebase. Settings are frozen, unknown `.env` keys are ignored, and secrets are
`SecretStr`.

**Follow-up: why the cached factory instead of a module-level instance?**

Import-time side effects. My previous version built the settings object and the
API clients at module scope, which meant importing a module to run one unit
test required a real OpenAI key. `lru_cache` gives singleton ergonomics but
defers the work to first use — and it is clearable in tests.

**Follow-up: how do you keep secrets out of logs?**

`SecretStr` renders as `**********` in `repr()` and tracebacks; you have to
call `.get_secret_value()` deliberately. That makes "do not log secrets" a
type-level property instead of a review convention.

**Follow-up: why no env-var prefix?**

`OPENAI_API_KEY` and the `LANGSMITH_*` variables are read directly from the
process environment by their own SDKs. A prefix would mean maintaining two
names for the same secret.

---

## Q4. Why standard-library logging?

**Answer.** Because everything in the dependency tree — LangChain, httpx,
qdrant-client, openai — logs through the stdlib. Configuring the stdlib root
logger means one handler captures their diagnostics as well as mine. With
`structlog` or `loguru` I would own my lines and lose theirs, which is exactly
the output I need when a retrieval call hangs.

I still get structure: a ~60-line JSON formatter that promotes `extra={...}`
fields to top-level keys.

**Follow-up: why pin third-party loggers to WARNING?**

`LOG_LEVEL=DEBUG` is something you set when you are already in trouble. If it
also turns on httpx request debugging, your own output disappears into the
noise.

**Follow-up: why is `configure_logging` idempotent?**

Repeated calls stacking handlers is the classic cause of every line printing
three times. It is guarded and tested.

---

## Q5. What is your testing strategy?

**Answer.** Unit tests with no network and no Docker, running in
milliseconds, plus integration tests behind a `pytest` marker so they are
deselectable. Unit tests use small handwritten HTML fixtures rather than the
8 MB Microsoft filing — a unit test should exercise the *shape* of a problem,
such as a nested inline-XBRL tag, not the size of it.

The subtle part is isolation. The developer's real `.env` sits at the repo
root and pydantic-settings would happily read it during a test, so
configuration tests use an `IsolatedSettings` subclass with `env_file=None`
plus a fixture that clears the relevant environment variables. Otherwise a test
passes on your laptop and fails in CI — or worse, passes for the wrong reason.

**Follow-up: what do you test in a RAG system that has no LLM yet?**

Contracts and invariants. Right now: that secrets never appear in `repr()`,
that invalid config fails fast, that logging is idempotent, that `extra=`
fields survive JSON serialisation. Once ingestion lands: metadata correctness,
chunk boundary behaviour, and that overlap actually overlaps. Retrieval quality
is measured by the eval harness, not by unit tests — those are different tools
for different questions.

---

## Q6. Why so few dependencies for a RAG project?

**Answer.** A package is added in the same commit as the module that imports
it. Right now the foundation needs `langchain-core`, `langchain-text-splitters`,
`langsmith`, `beautifulsoup4` and pydantic — so `qdrant-client`,
`langchain-openai`, `rank-bm25` and `cohere` are not installed yet.

In my previous version `fastapi` and `uvicorn` were installed for months before
a single endpoint existed. If a CVE had landed on either, triage would have
started with "do we even use this?" A dependency list should be a true
statement about the code.

---

## Q7. Why track 12 MB of filings in git?

**Answer.** Because the data is public, immutable, and small, and because the
alternative left the repo un-runnable from a clone — the filings existed on one
laptop. Tracking them also pins the corpus to the eval numbers it produced,
which is what makes historical results reproducible.

The anti-pattern is real at scale — git keeps every version forever — so the
binding constraint is corpus growth, not current size. If it grows, this gets
replaced by a download script with checksums, not by LFS.

---

## Q8. What is the weakest part of this right now?

**Answer.** It does not retrieve anything yet — the foundation is complete and
the pipeline is not. Beyond that, three honest weaknesses:

1. The one-way import rule is enforced by review, not tooling. With a second
   contributor I would add an `import-linter` contract to CI.
2. The eval set carried over is thin — 8 items, and one expected substring
   (`"net sales"`) is loose enough to produce false positives. That has to grow
   before any retrieval number means much.
3. Table extraction from SEC HTML is a known unsolved problem here: figures get
   flattened out of income statements. Retrieval hides it; generation will not.

Being able to name these is the point. A candidate who says "nothing" has not
looked.
