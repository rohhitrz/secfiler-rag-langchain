# Documentation Index

Read in this order the first time; jump directly by symptom afterwards.

## Understand the system

| Doc | Answers |
|---|---|
| [01 — High-Level Design](01-hld.md) | What is this, what are the components, what did we trade off |
| [02 — Low-Level Design](02-lld.md) | What is each module's contract (grows with the code) |
| [03 — Folder Structure](03-folder-structure.md) | Why each directory exists; where a new file goes |
| [04 — Data Flow](04-data-flow.md) | End-to-end: HTML → chunks → vectors → answer |
| [05 — Request Lifecycle](05-request-lifecycle.md) | What happens to one question, phase by phase |

## Work on the system

| Doc | Answers |
|---|---|
| [06 — Setup](06-setup.md) | Getting it running from a clone |
| [07 — Development](07-development.md) | How a module gets added; coding standards |
| [08 — Debugging](08-debugging.md) | The answer is wrong — now what |

## Reason about the system

| Doc | Answers |
|---|---|
| [09 — Failure Modes](09-failure-modes.md) | The five ways RAG fails and how to tell them apart |
| [10 — Scaling & Performance](10-scaling-and-performance.md) | Where the time goes; what breaks first at scale |
| [ADRs](adr/) | Every non-obvious decision, with rejected alternatives |
| [Interview prep](interview/) | Questions, answers, follow-ups per module |

## Conventions

- **HLD** describes the system; **LLD** describes contracts; **ADRs** describe
  decisions. If you are tempted to explain a decision inside the HLD, write an
  ADR and link it.
- ADRs are immutable. A decision that changes gets a new ADR that supersedes
  the old one — the history of *why* is the valuable part.
- Docs are written **with** the module, not after it. A module is not done
  until its LLD section, its ADR (if any) and its interview notes exist.
- Diagrams are Mermaid in Markdown, so they diff in review and never go stale
  in a binary file nobody can edit.
