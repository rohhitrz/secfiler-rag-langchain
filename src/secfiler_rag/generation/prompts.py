"""Prompt templates for grounded, cited answers.

Every instruction here exists because of a specific failure mode from
`docs/09-failure-modes.md`, not because it sounded good:

* **"Use only the context"** — the hallucination guard. Without it the model
  answers from parametric knowledge, which is especially dangerous here: it
  *has* read about Apple's revenue, so a plausible wrong figure is the likely
  output when the context is thin.
* **"If the context does not contain the answer, say so"** — the refusal path.
  A model asked to answer will answer. Making refusal an explicitly acceptable
  outcome is what makes "not in these filings" possible at all.
* **"Cite the bracketed number after each claim"** — traceability. Citations
  are what turn an answer into something a user can verify rather than trust.
* **"Quote figures exactly"** — filings are full of numbers, and a rounded or
  reformatted figure is a wrong figure in a financial context.
* **"State which company"** — three filings share heavy boilerplate, so an
  unattributed statement is ambiguous even when correct.

The refusal instruction is also enforced *outside* the prompt: an empty
retrieval never reaches the model at all (see `answer.py`). Prompt instructions
are guidance, not guarantees — the guarantee is code.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions \
about SEC 10-K filings for Apple (aapl), Microsoft (msft) and Tesla (tsla).

Rules:
1. Answer using ONLY the numbered context below. Never use outside knowledge, \
even if you are confident it is correct.
2. If the context does not contain the answer, say exactly what is missing. \
"The provided filings do not contain this information" is a correct and \
expected answer — never guess or infer a figure.
3. Cite the bracketed number of every block you used, immediately after the \
claim it supports, like [1] or [2][3].
4. Quote monetary amounts, percentages and dates exactly as they appear. Do \
not round, reformat, or convert units.
5. State which company a figure belongs to. The filings share boilerplate, so \
an unattributed number is ambiguous.
6. Be concise. Two or three sentences unless the question needs more."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer using only the context above, citing the bracketed numbers you used."""


def build_answer_prompt() -> ChatPromptTemplate:
    """Build the chat prompt for answer generation.

    A `ChatPromptTemplate` rather than string formatting because it keeps the
    system and user turns as separate messages — models weight system
    instructions differently from user content, and flattening them into one
    string discards that distinction.
    """
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
