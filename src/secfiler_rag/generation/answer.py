"""The "G" in RAG — retrieved context in, cited answer out.

**The refusal path is code, not a prompt instruction.** If retrieval returns
nothing, this module returns a refusal *without calling the model at all*. A
prompt saying "say so if you don't know" is guidance; a model handed an empty
context and an instruction to answer will still produce something fluent and
invented. The only reliable guarantee is never asking.

**What LCEL is doing.** `prompt | llm | parser` composes three `Runnable`s into
one. That gets `.invoke()`, `.batch()`, `.stream()` and automatic LangSmith
tracing for free, and it is genuinely worth using — but it is *plumbing*. The
decisions that determine answer quality are ours and live outside the chain:
which chunks are retrieved, how they are numbered, what the budget is, what the
prompt forbids, and whether the model is called at all.

**Citations are parsed and validated.** The model emits `[2]`; this module maps
that back to `(company, chunk_id)` and drops any marker that does not
correspond to a real block. A model citing `[7]` when four blocks were supplied
is hallucinating a source, and an unvalidated citation is worse than none — it
looks like provenance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from secfiler_rag.config import Settings, get_settings
from secfiler_rag.core.exceptions import ConfigurationError, GenerationError
from secfiler_rag.core.logging import get_logger
from secfiler_rag.generation.context import ContextBlock, build_context
from secfiler_rag.generation.prompts import build_answer_prompt
from secfiler_rag.retrieval.hybrid import Retriever

log = get_logger(__name__)

REFUSAL_MESSAGE = "The provided filings do not contain information to answer this question."

# Matches [1] and the [2][3] form the prompt asks for.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim's source, resolved from a marker the model emitted."""

    marker: int
    company: str
    chunk_id: int | None
    source: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A generated answer with everything needed to audit it."""

    question: str
    answer: str
    citations: tuple[Citation, ...] = ()
    context_blocks: tuple[ContextBlock, ...] = ()
    refused: bool = False
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def retrieved_documents(self) -> tuple[Document, ...]:
        """Documents that reached the prompt, in the order they were shown."""
        return tuple(block.document for block in self.context_blocks)


class AnswerGenerator:
    """Retrieve, ground, generate, and resolve citations."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        *,
        top_k: int = 3,
        max_context_tokens: int = 4000,
    ) -> None:
        """Wire the chain.

        Args:
            retriever: Any retrieval strategy. Generation does not care which,
                for the same reason the eval harness does not.
            llm: Any LangChain chat model.
            top_k: Chunks to retrieve and show the model.
            max_context_tokens: Ceiling for assembled context.
        """
        self._retriever = retriever
        self._top_k = top_k
        self._max_context_tokens = max_context_tokens
        # The LCEL chain: template -> model -> string. Plumbing, deliberately
        # thin; everything that decides answer quality happens around it.
        self._chain = build_answer_prompt() | llm | StrOutputParser()

    def answer(
        self,
        question: str,
        filters: Mapping[str, Any] | None = None,
        *,
        top_k: int | None = None,
    ) -> AnswerResult:
        """Answer a question from the filings.

        Args:
            question: The user's question.
            filters: Retrieval constraints, e.g. `{"company": "aapl"}`.
            top_k: Override the number of chunks shown to the model.

        Returns:
            The answer, its citations, and the exact context used.

        Raises:
            GenerationError: If the question is empty or the model call fails.
                Absence of an answer in the corpus is *not* an error.
        """
        if not question.strip():
            raise GenerationError("Cannot answer an empty question")

        k = top_k if top_k is not None else self._top_k
        documents = list(self._retriever.search(question, filters, k))

        if not documents:
            # The guardrail that actually holds: no context, no model call.
            log.info("refusing: no documents retrieved", extra={"question": question})
            return AnswerResult(question=question, answer=REFUSAL_MESSAGE, refused=True)

        blocks, context = build_context(documents, max_tokens=self._max_context_tokens)
        if not blocks:  # pragma: no cover - only if a single chunk exceeds the budget
            log.warning("refusing: context budget too small for any chunk")
            return AnswerResult(question=question, answer=REFUSAL_MESSAGE, refused=True)

        try:
            answer = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        citations = resolve_citations(answer, blocks)
        log.info(
            "generated answer",
            extra={
                "question": question,
                "blocks": len(blocks),
                "citations": len(citations),
                "answer_chars": len(answer),
            },
        )
        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            context_blocks=tuple(blocks),
            refused=_looks_like_refusal(answer),
        )


def resolve_citations(answer: str, blocks: Sequence[ContextBlock]) -> tuple[Citation, ...]:
    """Map bracketed markers in an answer back to their source chunks.

    Markers with no corresponding block are dropped: a model citing `[7]` when
    four blocks were supplied is inventing provenance, and an unvalidated
    citation is worse than none because it looks verifiable.

    Args:
        answer: The model's answer text.
        blocks: The context blocks that were shown.

    Returns:
        Unique citations, in the order first cited.
    """
    by_marker = {block.marker: block for block in blocks}
    seen: set[int] = set()
    citations = []

    for match in _CITATION_PATTERN.finditer(answer):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)

        block = by_marker.get(marker)
        if block is None:
            log.warning(
                "answer cited a block that was not provided",
                extra={"marker": marker, "blocks": len(blocks)},
            )
            continue

        citations.append(
            Citation(
                marker=marker,
                company=block.company,
                chunk_id=block.chunk_id,
                source=block.source,
            )
        )

    return tuple(citations)


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Build the configured chat model.

    Constructed on call, never at import — the same rule as every other client
    in this project.

    Raises:
        ConfigurationError: If `OPENAI_API_KEY` is not set.
    """
    settings = settings or get_settings()

    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set — required for answer generation. "
            "Add it to your .env file (see .env.example)."
        )

    return ChatOpenAI(
        model=settings.llm_model,
        # Zero by default: every claim must trace to a filing, and sampling
        # variety is the opposite of what a grounded answer needs.
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def _looks_like_refusal(answer: str) -> bool:
    """Heuristic flag for an answer that declined to answer.

    Advisory only — used for reporting, never to alter the answer. A precise
    signal would need structured output, which is a later refinement.
    """
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do not contain",
            "does not contain",
            "not contain information",
            "no information",
            "cannot be determined from",
        )
    )
