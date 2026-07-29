# ADR 0007 — Preserve inline-XBRL values; flatten tables into rows

**Status:** Accepted · **Date:** 2026-07-29

## Context

SEC filings are **inline-XBRL** documents. Machine-readable tags do not sit
*beside* the human-readable text — they *wrap* it:

```html
<p>Total net sales
   <ix:nonFraction name="us-gaap:Revenues" contextRef="FY2025" scale="6">416,161</ix:nonFraction>
</p>
```

The previous build stripped the entire `ix:` namespace with `decompose()`,
which removes a tag **and its contents**. The reasoning was sound — inline-XBRL
carries GAAP taxonomy URLs and period markers that pollute embeddings — but the
implementation deleted the financial data along with the metadata.

Measured on the Apple FY2025 filing:

| | `decompose()` (old) | `unwrap()` (new) |
|---|---|---|
| Clean text | 158,914 chars | 206,713 chars |
| Digits retained | 4,574 | 9,745 |
| Content spans lost | 251 | 0 |

Among the 251 deleted spans: `10-K`, `Apple Inc.`, `California 94-2404110`,
and the fiscal year end date. **53% of every digit in the document was being
discarded before it ever reached an embedding.**

This is the root cause of the `Products $ $ $` symptom carried in the previous
build's notes for four modules, and it had been misfiled as a *table* problem.

A second, separate issue: a 10-K is mostly tables. Flattening a table with a
plain text join emits every label, then every figure, so a row's label ends up
hundreds of characters from its numbers — retrievable, but unanswerable.

## Decision

**1. Unwrap inline-XBRL, do not decompose it.**

- `ix:header`, `ix:hidden`, `ix:references`, `ix:resources` → `decompose()`.
  These are containers whose contents genuinely are machine-only.
- Every other `ix:*` tag → `unwrap()`. The tag disappears; the value stays.
  Attributes never enter the text, so taxonomy names do not leak.

**2. Flatten each `<table>` into one line per `<tr>`, cells joined by ` | `.**

```
Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285
```

Empty spacer cells are dropped, and orphaned `$` / `%` cells are reattached to
their numbers.

**3. Preserve line structure; collapse only intra-line whitespace.**

The previous cleaner collapsed *all* whitespace including newlines. That
matters more than it sounds: a recursive splitter with no line or paragraph
boundaries to split on degenerates into blind character slicing.

**4. Write our own cleaner rather than using `BSHTMLLoader`.**

## Alternatives

**LangChain's `BSHTMLLoader`.** It is `BeautifulSoup.get_text()` with a file
read attached. It has no concept of inline XBRL and no table handling, so it
reproduces exactly the failure above. Using it would mean post-processing its
output — more work than owning the 120 lines.

**`UnstructuredHTMLLoader`.** Genuinely capable, with table and layout
inference. Rejected for now: a heavy dependency tree (and optional native
components), slower by an order of magnitude, and its behaviour on filings is
opaque enough that debugging a lost figure would be harder than the problem it
solves. A reasonable measured upgrade later.

**Convert tables to Markdown or HTML in the chunk text.** Markdown pipe tables
are close to what we emit, minus the header/alignment rows that cost tokens and
add nothing to an embedding. Keeping raw HTML would poison both embeddings and
BM25 with tag noise.

**Keep the XBRL attributes as chunk metadata.** Genuinely interesting — the
`us-gaap:Revenues` name is a precise semantic label, and a future
structured-retrieval path could exploit it. Deferred: it changes the metadata
contract and buys nothing until there is a retriever that uses it.

## Consequences

- Chunk count rises from 768 to **1,309** across the corpus (+70%), because
  ~30% more real text now survives. Indexing costs more; the recovered content
  includes the figures the system exists to answer questions about.
- Retained inline-XBRL means some fragments (`false`, `P1Y`, scale markers)
  from `ix:nonNumeric` tags still reach the text. Lower-cost noise than
  deleting revenue.
- Table rows are longer lines, so the splitter's `"\n"` separator does real
  work — a row stays intact instead of being cut through the middle of a set of
  figures.
- We own ~120 lines of cleaning logic. It is tested against both miniature
  fixtures and the real corpus, and the integration test asserts on actual
  FY2025 figures so a regression here fails loudly.

## Interview angle

> **Q: What was the hardest bug in this project?**
>
> A silent one. My first version stripped the inline-XBRL namespace from SEC
> filings, which sounds obviously right — it is machine metadata. But inline
> XBRL *wraps* the visible values rather than sitting beside them, so removing
> the tags with their contents deleted 53% of the digits in the document,
> including the company name and the form type. Nothing raised. Retrieval kept
> working, evals kept passing, and the income statement quietly read
> `Products $ $ $`.
>
> I had it misfiled as a table-extraction problem for weeks. What found it was
> diffing the two cleaning strategies word by word and reading what disappeared
> — the fix was `unwrap()` instead of `decompose()`, one line, once I knew
> which line.
>
> **Follow-up: how do you stop it regressing?**
>
> An integration test asserts on real FY2025 figures from the actual filing —
> total net sales `416,161`, `Apple Inc.`, `10-K`. Fixture-only tests would not
> have caught it, because I would have written the fixture with the same wrong
> mental model that caused the bug.
>
> **Follow-up: why not use LangChain's HTML loader?**
>
> `BSHTMLLoader` is `get_text()` with a file read. It would reproduce exactly
> this bug. This is the part of the pipeline where domain knowledge lives, so
> it is the part worth owning — I use LangChain for the splitter, where the
> generic algorithm is genuinely better than what I would write.
