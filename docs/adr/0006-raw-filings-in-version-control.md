# ADR 0006 — Track raw filings in version control

**Status:** Accepted · **Date:** 2026-07-29

## Context

The corpus is three SEC 10-K HTML files, ~12 MB total:

| File | Size |
|---|---|
| `aapl-2025.htm` | 1.5 MB |
| `msft-2025.htm` | 8.2 MB |
| `tsla-2025.htm` | 2.4 MB |

The previous build git-ignored `data/raw/`, so the repository was not runnable
from a clone — the filings existed on exactly one laptop. For a portfolio
project that reviewers are expected to clone and run, that is a real defect.
It is also a single-point-of-failure for the only genuinely irreplaceable
asset in the project: SEC filings get superseded, and "the FY2025 10-K as it
existed on this date" is not trivially re-fetchable.

## Decision

Track `data/raw/*.htm` in git. Everything derived from them —
`qdrant_storage/`, cleaned text, chunk dumps — stays ignored.

`.gitignore` states this as an explicit exception:

```gitignore
data/*
!data/.gitkeep
!data/raw/
!data/raw/**
qdrant_storage/
```

## Alternatives

**A download script fetching from SEC EDGAR.** The purist answer, and correct
at larger scale. Rejected because: EDGAR URLs and document structure change,
the script becomes a maintenance surface and a network dependency for `pytest`,
and a reviewer cloning the repo hits an unnecessary failure mode. Worse, it
would not pin *which* version of the filing produced the eval numbers.

**Git LFS.** Rejected: real friction (contributors need `git lfs` installed,
some hosts meter bandwidth) for a total under 15 MB. LFS earns its keep at
hundreds of megabytes, not twelve.

**External storage (S3 / a release asset).** Rejected as infrastructure for a
problem this size, and it reintroduces the "not runnable from a clone" defect.

## Consequences

- Clone size ~12 MB. Acceptable; well under GitHub's 100 MB per-file hard limit
  and its 50 MB warning threshold.
- The corpus is version-pinned alongside the eval numbers it produced, which is
  what makes historical results reproducible.
- **The binding constraint is corpus growth, not the current size.** If the
  corpus grows to dozens of filings, this decision gets superseded by a
  download script with a manifest of checksums — the checksums being the part
  that preserves reproducibility.
- These are public regulatory documents, so there is no licensing or privacy
  concern in redistributing them.

## Interview angle

> **Q: Data in git? Isn't that an anti-pattern?**
>
> It is at scale, and the reason is real: git stores every version forever, so
> churning binaries make the repo unclonable. Here the data is 12 MB, public,
> and immutable — a filed 10-K never changes. What I get in exchange is that
> the repo is runnable from a clone and that eval numbers are pinned to the
> exact corpus that produced them. I would reverse it the moment the corpus
> grows or starts changing, and the replacement is a download script with
> checksums, not LFS.
