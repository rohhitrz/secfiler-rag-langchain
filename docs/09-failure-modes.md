# Failure Modes

A taxonomy of how RAG systems fail, with the specific form each takes in this
project. This doubles as an interview framework: when asked "how would you
debug a RAG system", answering with five named failure classes and their
distinct signatures is a senior answer; "check the prompt" is not.

---

## The five classes

| # | Class | One-line signature |
|---|---|---|
| 1 | **Bad chunking** | The right document is indexed but no single chunk contains the answer |
| 2 | **Embedding / vocabulary mismatch** | Query and chunk mean the same thing but sit far apart in vector space |
| 3 | **Retrieval noise** | The right chunk is retrieved but buried under plausible-looking wrong ones |
| 4 | **Context overflow** | The right chunk is in the prompt but the model does not use it |
| 5 | **Hallucination** | The model asserts something the context does not support |

They are ordered by pipeline position. **Diagnose in order** — a chunking bug
looks exactly like a hallucination bug if you start at the end.

---

## 1. Bad chunking

**How it shows up here:** SEC filings are dominated by tables. Naive HTML text
extraction flattens an income statement into `Products $ $ $` — labels survive,
figures do not. Retrieval finds the "right" chunk and the number is simply
gone.

Also: a fixed 1000-character window can split "total net sales were" from the
figure that follows it. Neither half answers the question.

**Detection:** print the chunk that should contain the fact and read it.

**Mitigations, cheapest first:**
- Overlap (already baseline) — survives sentence-level splits
- Structure-aware splitting on headings/sections rather than character counts
- Dedicated table extraction that emits rows as text (`Product | 2025 | 2024`)
- Contextual retrieval: prepend a generated one-line summary of the parent
  section to each chunk

**Trade-off:** every improvement raises ingestion cost and complexity. Fixed-size
chunking is the honest baseline; the others must earn their place with a
measured lift.

---

## 2. Embedding / vocabulary mismatch

**How it shows up here:** the user asks "what was Apple's revenue?"; the filing
says "net sales". Or the user says "profit" and the filing says "operating
income". Dense retrieval usually handles synonyms — but financial language has
*near*-synonyms that are technically different line items, which is worse than
plain mismatch.

**Detection:** dense misses, BM25 finds it (or vice versa). See the
disagreement table in [`08-debugging.md`](08-debugging.md#24-dense-vs-sparse-disagreement).

**Mitigations:**
- Hybrid retrieval — BM25 catches exact terms ("Megapack", "Powerwall") that
  embeddings blur; vectors catch paraphrase. Neither alone is enough
- Query expansion / multi-query
- A finance-tuned embedding model — a measured upgrade, not an assumption

---

## 3. Retrieval noise

**How it shows up here:** three companies' filings share boilerplate (risk
factors, accounting policies). A query about "risk factors" matches all three
at nearly identical scores.

**Detection:** the correct chunk is in the candidate list but not the top-k.

**Mitigations:**
- Metadata filtering — scope to `company` when the question names one. Cheapest
  and highest-impact
- Reranking — a cross-encoder reads query and document *together*, which is
  strictly more informative than comparing two independently-computed vectors
- Wider candidate pools before the reranker

**Trade-off:** filtering is nearly free but requires knowing the company;
reranking costs an API call per query and adds 100–400 ms.

---

## 4. Context overflow / lost in the middle

**How it shows up here:** stuffing 10 chunks of dense financial prose into the
prompt. Models attend most reliably to the beginning and end of context;
material in the middle gets under-weighted even when it is present.

**Detection:** the answer is wrong, the correct chunk is provably in the
prompt, and it sits in the middle.

**Mitigations:**
- Send fewer, better chunks (3 after reranking, not 10 after fusion)
- Order by relevance so the strongest context is first
- Budget tokens explicitly rather than hoping

**This is the real argument for reranking.** Not "better ordering is nice" —
*fewer chunks that are actually relevant beats more chunks that might be*.

---

## 5. Hallucination

**How it shows up here:** the most dangerous class, because financial answers
look authoritative. The model produces a plausible revenue figure when the
retrieved chunk had its numbers stripped by class 1.

**Detection:** every claim must be traceable to a retrieved chunk. If it is not
in the context, it is a hallucination regardless of whether it happens to be
true.

**Mitigations:**
- Grounding instruction: answer *only* from context
- An explicit refusal path — "not found in these filings" must be an acceptable
  answer, and an empty retrieval must never reach the model with an instruction
  to answer anyway
- Citations per claim, tied to `chunk_id`
- Faithfulness evaluation (LLM-as-judge) as a measured metric

---

## Project-specific known issues

| Issue | Class | Status |
|---|---|---|
| Table figures lost in HTML extraction (`Products $ $ $`) | 1 | Known; to confront in ingestion + generation |
| Shared boilerplate across three filings | 3 | Mitigated by company filter |
| `"net sales"` as an eval substring is loose | — (eval design) | Tighten when the eval set grows |
| Cross-company comparison ("compare Apple and Tesla") | 3 | Parked — needs multi-filter retrieval and a different metric |

---

## Interview framing

> **Q: A user says the answer is wrong. Walk me through your process.**
>
> First I check whether the correct chunk was in the retrieved context — that
> single bisection splits the problem into retrieval-side and generation-side
> and eliminates half the search space in one step. If retrieval missed, I ask
> whether the chunk exists in the index at all, then whether a filter excluded
> it, then whether dense and sparse disagree — that tells me chunking versus
> vocabulary mismatch. If retrieval was fine, I print the rendered prompt and
> check position and grounding instructions. I do not touch the prompt until I
> have confirmed the context was correct, because prompt-tuning a retrieval bug
> is the most common way to waste a day.
