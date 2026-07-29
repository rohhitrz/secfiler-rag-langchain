# ADR 0009 — Deterministic point IDs and a single filtered collection

**Status:** Accepted · **Date:** 2026-07-29

## Context

Two write-path decisions that are cheap to make correctly now and expensive to
change once a collection is populated.

**Point identity.** Every vector-store quickstart auto-generates point IDs. That
makes the indexer *append-only*: running it twice stores every chunk twice.
There is no error — the second run succeeds — and the symptom appears later as
gradual retrieval degradation, because duplicates crowd the top-k and the
reranker receives three copies of the same passage instead of three distinct
candidates.

The previous build hit a sharper version of this. It used `id=i` (the chunk's
position), which collided across companies: every filing restarts at 0, so
Tesla's chunk 0 overwrote Apple's. Points silently disappeared.

**Corpus scoping.** Three companies share heavy boilerplate — risk factors,
accounting policies, forward-looking-statement disclaimers. A query about
"risk factors" matches all three at nearly identical scores, so retrieval needs
a way to scope to one company.

## Decision

**1. Point IDs are `uuid5(NAMESPACE_DNS, f"{company}-{chunk_id}")`.**

Deterministic, so re-indexing overwrites in place. Two properties of the key:

- **`company` is part of it** — `chunk_id` restarts per filing, so it is
  ambiguous alone. Identity is the *pair*.
- **The separator is load-bearing** — without it, `("aapl1", 2)` and
  `("aapl", 12)` both produce `aapl12`.

`uuid5` (not `uuid4`) because it is a hash of the name: same input, same output,
forever, on any machine. Qdrant accepts only UUIDs or unsigned integers as point
IDs, so a natural string key has to be hashed into one.

**2. One collection (`filings`) for all companies, scoped by payload filter.**

Not one collection per company.

**3. The collection is created explicitly by us, not auto-created on write.**

`ensure_collection` declares vector size and distance, and **verifies** an
existing collection's vector size, raising `IndexingError` on mismatch.

**4. A keyword payload index on `metadata.company`.**

Note the prefix. `QdrantVectorStore` nests document metadata under a `metadata`
payload key rather than storing it at the root:

```json
{"page_content": "...", "metadata": {"company": "aapl", "chunk_id": 12}}
```

Every filter and index must therefore address `metadata.company`. This is
captured once as `COMPANY_PAYLOAD_FIELD` and asserted in a test, because
filtering on the bare field name silently matches nothing.

## Alternatives

**Auto-generated IDs, drop the collection before each re-index.** Works, but
makes re-indexing destructive: there is a window where the collection is empty
and any concurrent reader gets nothing. Deterministic IDs give upsert semantics
with no such window.

**A hash of the chunk text as the ID.** Content-addressed, so identical
boilerplate deduplicates automatically — appealing until you realise it also
*loses* the second occurrence, and "the same paragraph in Apple's and Tesla's
filings" is two legitimately distinct results. Rejected: dedup is a retrieval
concern, not an identity one.

**Per-company collections.** Hard isolation, and cross-company queries become
impossible without fanning out and merging manually. The payload filter is a
*soft, query-time* boundary: one query can scope to a company, widen to several,
or drop the filter entirely. Per-collection becomes right when tenants need real
isolation — a privacy decision, not a performance one.

**Let `QdrantVectorStore` auto-create the collection.** It infers vector size by
embedding a probe string. That makes the schema a side effect of whichever code
path ran first, and it silently writes into a collection created last month with
different settings. Explicit creation turns that into a startup error.

**No payload index.** Qdrant would apply the filter after the vector search, so
a company-scoped query still pays to search every other company's vectors — and
recall suffers, because the candidate pool it filtered was shared across all
three.

## Consequences

- Re-indexing is idempotent and safe to run repeatedly; the CLI needs no
  "delete first" step, and `--recreate` exists only for a deliberate schema
  change.
- Changing chunk size renumbers every `chunk_id`, so IDs change and stale points
  from the old configuration linger. `--recreate` is the answer, and this is
  worth remembering before tuning the chunker.
- The metadata payload nesting is a hard dependency on `QdrantVectorStore`'s
  layout. Isolated behind one constant so a change is a one-line edit.
- Cross-company retrieval remains possible later via `should` / `MatchAny`
  without re-indexing anything.

## Interview angle

> **Q: How do you handle re-indexing?**
>
> Point IDs are a UUID5 hash of `(company, chunk_id)`, so re-running the indexer
> upserts in place instead of appending. That matters because the failure mode
> of auto-generated IDs is silent: the second run succeeds, the corpus doubles,
> and you only notice weeks later when every result set contains the same
> passage three times.
>
> My first version used the chunk's position as the ID, which collided across
> companies — every filing restarts at 0, so Tesla's chunk 0 overwrote Apple's.
> That is why identity is the pair, not the integer, and why the separator in
> the key is load-bearing.
>
> **Follow-up: why one collection instead of one per company?**
>
> A payload filter is a soft boundary applied at query time, so one query can
> scope to a company, widen to several, or drop the filter. Per-company
> collections are a hard storage boundary — cross-company questions would mean
> fanning out and merging rankings myself. I would switch if tenants needed real
> isolation, but that is a privacy requirement, not a retrieval one.
>
> **Follow-up: anything surprising in the integration?**
>
> Yes — LangChain's Qdrant store does not put your metadata at the payload root.
> It nests it under a `metadata` key, so filters have to address
> `metadata.company`. Filtering on `company` matches nothing and raises nothing.
> I pinned it as a single constant and wrote a test asserting its value, because
> that is exactly the kind of framework detail that silently breaks a filter
> during a refactor.
