# Evaluation Datasets

Eval data is **measurement infrastructure**, so it is version controlled and
lives outside the package: changing a dataset invalidates every number produced
before the change, and that must be visible in a diff.

```
evals/datasets/
└── seed_eval_set.json    # 8 hand-written pairs carried over from the first build
```

The *harness* that consumes these lives in
`src/secfiler_rag/evaluation/` — data and code are deliberately separate so a
dataset can be swapped without touching a line of Python.

## Format

```json
{"tier": 2, "query": "...", "expected_substring": "...", "company": "aapl"}
```

**Why a substring rather than a chunk ID?** Chunk IDs change every time the
chunking configuration changes, which would silently invalidate the whole set
on any ingestion tweak. A substring of the cleaned text survives re-chunking,
so the same dataset can compare a 1000-character chunker against an 800 one.

**Why two tiers?**

- **Tier 1** — near-tautological lexical queries. These are not a measure of
  retrieval quality; they are a smoke test. A Tier 1 failure means the index or
  the harness is broken.
- **Tier 2** — realistic natural-language questions with different vocabulary
  from the filing text. This is the number that should actually move when
  retrieval improves.

## Rules

1. **Write expected substrings from the cleaned pipeline output**, never from
   the filing as rendered in a browser. HTML extraction changes whitespace and
   drops table structure, so a substring copied from a browser can be
   unmatchable even when retrieval is perfect.
2. **Audit every pass.** A loose substring can match dozens of chunks and
   report a hit while retrieval actually missed. Two such false positives were
   caught in the previous build.
3. **Never edit the set to make a number go up.** If an item is wrong, fix it
   and re-baseline every recorded score — and say so in `PROGRESS.md`.
4. **Version the file** when its meaning changes, so old numbers stay
   attributable to the dataset that produced them.
