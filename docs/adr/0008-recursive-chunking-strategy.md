# ADR 0008 — Recursive character chunking at 1000/200

**Status:** Accepted · **Date:** 2026-07-29

## Context

Chunking is where retrieval quality is actually decided. A chunk is the atomic
unit of retrieval: too large and the embedding averages several topics into a
vector that matches nothing precisely; too small and the chunk loses the
context that makes it interpretable. No reranker recovers a fact that was split
across two chunks, and no prompt repairs one.

The previous build sliced every 1000 characters unconditionally, with 200
characters of overlap. That is the honest baseline — it is also blind to
sentence and table-row boundaries, so it regularly cut through the middle of a
row of figures.

## Decision

Use LangChain's **`RecursiveCharacterTextSplitter`** with:

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 1000 chars | Carried over as the measured baseline |
| `chunk_overlap` | 200 chars (20%) | A fact on a boundary survives in one neighbour |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Structure first, blind cut last |
| `add_start_index` | `True` | Auditability: where in the filing did this come from |

The separator list is the decision that matters. The splitter tries each in
order and only falls back to a character cut when nothing else fits — so
`"\n"` keeps a flattened table row intact, which is only possible because the
cleaner deliberately preserves line structure (see
[ADR 0007](0007-preserve-inline-xbrl-values.md)).

Size and overlap live in `Settings`, with a validator rejecting
`overlap >= size` — at that point the stride is zero and the splitter stops
advancing through the text.

Metadata on every chunk: `company`, `chunk_id`, `source`, `start_index`.
Identity is the **pair** `(company, chunk_id)`, never `chunk_id` alone, because
IDs restart at 0 per filing.

## Alternatives

**Fixed-size character splitting** (the previous approach). Rejected: cuts
sentences and table rows mid-way for no benefit. The recursive splitter
degrades to exactly this when no separator matches, so it is strictly better.

**Token-based splitting** (`TokenTextSplitter`, tiktoken). Genuinely more
correct — embedding models have *token* limits, not character limits, and the
character/token ratio varies with content (a table of figures tokenises very
differently from prose). Deferred, not rejected: it adds a tokeniser dependency
and 1000 characters is comfortably inside `text-embedding-3-small`'s 8191-token
window, so nothing is being truncated. This is the first upgrade to measure.

**Structure-aware splitting** on filing sections (Item 1A Risk Factors, Item 7
MD&A). Attractive — those sections are the natural retrieval unit and the
headings are excellent metadata. Rejected for now because SEC HTML marks
sections by styling convention rather than semantic tags, so detection is
heuristic and filer-specific. Worth revisiting with a measured lift.

**Semantic chunking** (split where embedding similarity drops). Rejected as
premature: it costs an embedding call per sentence at ingestion time, and it
should be evaluated against a stable baseline rather than adopted as one.

**Contextual retrieval** (prepend an LLM-generated summary of the parent
section to each chunk). Strong technique with published gains. Explicitly
scheduled *last*, because a measured lift is only meaningful against a stable
core.

## Consequences

- 1,309 chunks across the corpus (aapl 292, msft 441, tsla 576).
- ~20% storage and embedding overhead from overlap. Accepted: boundary loss is
  silent and unrecoverable, storage is cheap and visible.
- Character-based sizing means chunk *token* counts vary. Fine at 1000 chars;
  revisit before raising the size.
- `chunk_id` is stable only for a given chunker configuration. Changing size or
  overlap renumbers everything — which is exactly why the eval set keys on text
  substrings rather than chunk IDs.

## Interview angle

> **Q: How did you choose your chunk size?**
>
> I inherited 1000/200 as a baseline and have not yet earned the right to
> change it — I have no retrieval numbers on this corpus, so tuning it now
> would be guessing with extra steps. What I did change is *how* it splits:
> recursive on paragraph, then line, then sentence, so a flattened table row
> stays intact instead of being cut through the middle of a set of figures.
>
> The overlap is the part worth explaining. 200 of 1000 characters exist purely
> so a fact that lands on a boundary survives in at least one neighbour —
> because if it does not, neither chunk can answer the question and no
> downstream component can fix it.
>
> **Follow-up: characters or tokens?**
>
> Characters today, and that is a known weakness — embedding models have token
> limits, and prose tokenises very differently from a table of figures. It is
> safe at 1000 characters because that is far inside the model's 8191-token
> window, so nothing truncates. Token-based splitting is the first upgrade I
> would measure if I raised the chunk size.
>
> **Follow-up: what would you try next to improve retrieval?**
>
> Section-aware chunking, because Item 1A and Item 7 are the natural retrieval
> units and the headings are good metadata. But I would want an eval number
> first — the previous build taught me that structural changes to chunking
> renumber every chunk, so if I cannot measure the before and after, I cannot
> tell an improvement from a reshuffle.
