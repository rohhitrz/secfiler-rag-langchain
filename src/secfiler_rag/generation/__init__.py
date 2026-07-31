"""Stage 4 — the "G" in RAG: retrieved context in, cited answer out.

Responsibility: prompt templates, context assembly with token budgeting, the
answer chain, and citation resolution back to `(company, chunk_id)`.

Two guarantees that are code rather than prompt text:

* **Refusal.** Empty retrieval never reaches the model. A model handed no
  context and told to answer will still produce something fluent and invented.
* **Citation validity.** Markers the model emits are resolved against the
  blocks actually supplied; anything else is dropped. An unvalidated citation
  is worse than none, because it looks like provenance.
"""

from secfiler_rag.generation.answer import (
    REFUSAL_MESSAGE,
    AnswerGenerator,
    AnswerResult,
    Citation,
    build_llm,
    resolve_citations,
)
from secfiler_rag.generation.context import ContextBlock, build_context, count_tokens
from secfiler_rag.generation.prompts import build_answer_prompt

__all__ = [
    "REFUSAL_MESSAGE",
    "AnswerGenerator",
    "AnswerResult",
    "Citation",
    "ContextBlock",
    "build_answer_prompt",
    "build_context",
    "build_llm",
    "count_tokens",
    "resolve_citations",
]
