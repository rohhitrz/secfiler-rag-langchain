# Interview — Module 1: Ingestion

The module with the best story in the project, because it contains a real bug
that was found by measurement rather than by reading code.

---

## Q1. Walk me through your ingestion pipeline.

**Answer.** Four stages, each its own module. `loader` resolves a filing's
identity from its filename and reads the bytes. `cleaner` turns SEC HTML into
retrievable plain text. `splitter` chunks that text into LangChain `Document`s
with metadata. `pipeline` composes the three.

The split exists so the middle two stages are pure string-to-string functions —
no filesystem, no config. That makes them trivially testable and reusable from
a notebook, and it means only one module in the package touches both disk and
settings.

**Follow-up: why enforce a filename convention instead of passing the company
in?**

Because the company key is load-bearing all the way to the Qdrant payload
filter, and the failure mode when it is wrong is silent. `AAPL-2025.htm` would
produce an uppercase key that never matches a lowercase filter — you get zero
results and no error. So a filename that violates the convention raises at
ingestion instead of producing a system that looks fine and retrieves nothing.

---

## Q2. What was the hardest bug in this project?

**This is the answer to lead with.**

My first version stripped the inline-XBRL namespace from SEC filings, which
sounds obviously correct — it is machine metadata full of GAAP taxonomy URLs.
But inline XBRL *wraps* the visible values rather than sitting beside them:

```html
<ix:nonFraction name="us-gaap:Revenues" scale="6">416,161</ix:nonFraction>
```

`decompose()` removes a tag **and its contents**, so I was deleting the revenue
figure along with the tag. Measured on Apple's FY2025 filing: 53% of every
digit in the document, plus 251 spans of real content including `Apple Inc.`,
the form type `10-K`, and the fiscal year end date.

Nothing raised. Retrieval kept working. Evals kept passing. The income
statement quietly read `Products $ $ $`, and I had it misfiled as a
table-extraction problem for weeks.

What found it was diffing the two cleaning strategies word by word and reading
what disappeared. The fix was `unwrap()` instead of `decompose()` — one line,
once I knew which line.

**Follow-up: how do you prevent it regressing?**

An integration test asserts on real figures from the actual filing — total net
sales `416,161`, `Apple Inc.`, `10-K`. A fixture-only test would not have
caught it, because I would have written the fixture with the same wrong mental
model that caused the bug in the first place.

**Follow-up: what is the general lesson?**

Silent data loss in preprocessing is the most expensive class of RAG bug,
because every symptom appears somewhere else. The retrieval metrics looked
fine. The chunk looked plausible. The failure only became visible at
generation, four stages downstream from the cause.

---

## Q3. How do you handle tables? A 10-K is mostly tables.

**Answer.** Each `<table>` becomes one line per `<tr>`, cells joined with
` | `:

```
Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285
```

The point is adjacency. Naive text extraction emits all the labels, then all
the figures, so a row's label ends up hundreds of characters from its numbers —
retrievable, but unanswerable, because the model cannot tell which figure
belongs to which line item.

Two details: empty cells are dropped, since filings use them heavily for visual
alignment, and orphaned `$` and `%` cells are reattached to their numbers, so
`$ | 416,161 | 6 | %` becomes `$416,161 | 6%`.

**Follow-up: why not Markdown tables?**

Close to what I emit, minus the header and alignment rows — those cost tokens
and add nothing to an embedding. Keeping the raw HTML would poison both the
embeddings and BM25 with tag noise.

**Follow-up: what about nested tables?**

Tables are flattened innermost-first, so a nested table is already text by the
time its parent is processed. The current filings do not nest tables — I
checked, 62 tables, zero nested — but the ordering costs nothing and removes a
class of silent corruption.

---

## Q4. Why LangChain's splitter but your own cleaner?

**Answer.** This is the line I draw throughout the project: **use the framework
where the generic algorithm is genuinely better than mine; own the parts where
domain knowledge lives.**

`RecursiveCharacterTextSplitter` is a good generic algorithm — try paragraph,
then line, then sentence, then word, and only fall back to a blind character
cut when nothing else fits. I would write something worse.

`BSHTMLLoader` is `BeautifulSoup.get_text()` with a file read attached. It has
no concept of inline XBRL and no table handling, so it reproduces exactly the
bug in Q2. Domain knowledge about SEC filings is the value in this module, so
it is the part worth owning.

**Follow-up: what about `UnstructuredHTMLLoader`?**

Genuinely capable, with layout and table inference. I rejected it for now on
three grounds: a heavy dependency tree, an order of magnitude slower, and
opaque enough that debugging a lost figure would be harder than the problem it
solves. It is a reasonable measured upgrade later — but I would want a
retrieval number to justify it, not a feature list.

---

## Q5. How did you choose chunk size and overlap?

**Answer.** 1000 characters with 200 overlap, carried over as a baseline. I
have not earned the right to change it — I have no retrieval numbers on this
corpus yet, so tuning it now would be guessing with extra steps.

The overlap is the part worth explaining: 200 of every 1000 characters exist
purely so a fact landing on a boundary survives in at least one neighbour. If
it does not, neither chunk can answer the question, and no reranker or prompt
downstream can repair it. The cost is about 20% more chunks.

**Follow-up: characters or tokens?**

Characters, and that is a known weakness — embedding models have token limits,
and prose tokenises very differently from a table of figures. It is safe at
1000 characters because that is far inside `text-embedding-3-small`'s
8191-token window, so nothing truncates. Token-based splitting is the first
thing I would measure if I raised the size.

**Follow-up: what is `start_index` for?**

Auditability. When a retrieved chunk looks wrong, `start_index` tells me
exactly where in the cleaned filing it came from, so I can read the
surrounding text and see whether the problem is chunking or retrieval. It costs
one integer per chunk.

**Follow-up: what would you try next?**

Section-aware chunking on Item 1A and Item 7 — those are the natural retrieval
units and the headings are good metadata. But SEC HTML marks sections by
styling convention rather than semantic tags, so detection is heuristic and
filer-specific. I would want an eval number before taking on that complexity.

---

## Q6. Why does one bad filing fail the whole ingestion run?

**Answer.** Because partial ingestion is worse than a loud failure. If Microsoft
silently fails to index, the system still starts, still answers Apple questions
correctly, and returns confident wrong answers about Microsoft — a retrieval
gap that surfaces weeks later as an unexplained quality complaint.

There is one deliberate exception: undecodable bytes are replaced rather than
raised. Losing one stray character is better than failing an entire 8 MB
document.

---

## Q7. What are the numbers?

| Company | Raw | Clean text | Chunks |
|---|---|---|---|
| aapl | 1.5 MB | 209,393 chars | 292 |
| msft | 8.2 MB | 317,163 chars | 441 |
| tsla | 2.4 MB | 399,145 chars | 576 |
| **Total** | **12.1 MB** | | **1,309** |

The previous build produced 768 chunks from the same corpus. The 70% increase
is recovered content, not smaller chunks — it is the data the old cleaner was
deleting.

Worth noting the compression ratio too: Microsoft's filing is 96% markup by
bytes. That is not a curiosity, it is why cleaning quality dominates everything
downstream.

---

## Q8. What is still weak here?

1. **Character-based chunking**, as above — a token-based splitter is more
   correct.
2. **`ix:nonNumeric` fragments** still leak short tokens like `false` and `P1Y`
   into the text. Low-cost noise compared to deleting revenue, but noise.
3. **No section awareness** — a chunk cannot currently say "this came from Item
   1A Risk Factors", which is metadata a filter would love to have.
4. **No deduplication.** Boilerplate repeats within and across filings, so
   near-duplicate chunks compete in retrieval.

None of these is worth fixing before there is an eval number to measure the fix
against.
