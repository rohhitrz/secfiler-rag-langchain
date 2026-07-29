# Interview Preparation

One file per module, written **when that module lands**, while the reasoning is
fresh. Each file follows the same shape:

1. **Questions you will be asked** — the obvious ones and the sharp ones
2. **Expected answer** — how to answer well, not a script to recite
3. **Follow-ups** — where a good interviewer pushes next
4. **Trade-offs** — what you gave up
5. **Alternatives** — and when they would be the better call

## Index

| Doc | Covers |
|---|---|
| [00 — Foundation](00-foundation.md) | Repo structure, config, logging, testing, dependency policy |
| _01 — Ingestion_ | (with Module 1) |
| _02 — Indexing_ | (with Module 2) |
| _03 — Retrieval & evaluation_ | (with Module 3) |
| _04 — Hybrid & reranking_ | (with Modules 4–5) |
| _05 — Generation_ | (with Module 6) |

## How to use these

**Do not memorise answers.** Interviewers detect recitation instantly, and the
follow-up question breaks it. Use these to check that you can reconstruct the
*reasoning* — if you can explain why the alternative was rejected, you can
handle a question phrased in a way this document did not anticipate.

**The strongest answer names a trade-off you accepted.** "We chose X" is weak.
"We chose X over Y because Z, and the cost is W, which we accepted because V"
is what distinguishes someone who made a decision from someone who followed a
tutorial.

**Rule from the working agreement:** never keep a line of code you cannot
explain. If a snippet in this repo would embarrass you under questioning, that
is a signal to go read it, not to hope it does not come up.

## Cross-cutting questions

These come up regardless of which module is being discussed:

- *Walk me through what happens when a user asks a question.* →
  [`05-request-lifecycle.md`](../05-request-lifecycle.md)
- *How do you know your RAG is any good?* → the eval harness, and the fact that
  it is retriever-agnostic
- *A user says the answer is wrong. What do you do?* →
  [`09-failure-modes.md`](../09-failure-modes.md), the retrieval/generation
  bisection first
- *What breaks at 100× scale?* →
  [`10-scaling-and-performance.md`](../10-scaling-and-performance.md), and know
  which constraint binds first
- *What would you do differently?* → have a real answer; every ADR's
  "Consequences" section is a candidate
