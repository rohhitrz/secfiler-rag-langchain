"""Assemble retrieved chunks into promptable, citable context.

**Why context blocks are numbered.** A model cannot cite `(aapl, chunk_127)`
reliably — it is a long opaque token it has no reason to reproduce exactly. It
can reliably emit `[2]`. So each chunk is labelled with a small integer, the
model is instructed to cite those integers, and this module owns the mapping
back to `(company, chunk_id)`. The model never sees a chunk ID and never has to
get one right.

**Why the token budget is explicit.** Context that overflows the model's window
is truncated *by the provider*, silently, from the end — so the chunk retrieval
worked hardest to find is exactly the one that disappears. Counting tokens and
dropping whole chunks means the loss is deliberate, ordered worst-first, and
logged.

Chunks are dropped from the *end* because they arrive ranked: after reranking,
position 1 is the cross-encoder's best judgement. If something must go, it
should be the weakest evidence, not the strongest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document

from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# cl100k_base covers the GPT-4 family. An exact match matters less than
# consistency: this is a budget, and a stable approximation is enough.
_ENCODING = "cl100k_base"

# Rough allowance for the block header we add around each chunk.
_BLOCK_OVERHEAD_TOKENS = 24


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One numbered piece of evidence handed to the model."""

    marker: int
    document: Document

    @property
    def company(self) -> str:
        """Company this evidence came from."""
        value = self.document.metadata.get("company")
        return value if isinstance(value, str) else "unknown"

    @property
    def chunk_id(self) -> int | None:
        """Chunk identifier, for tracing a citation back to the corpus."""
        value = self.document.metadata.get("chunk_id")
        return value if isinstance(value, int) else None

    @property
    def source(self) -> str:
        """Source filename."""
        value = self.document.metadata.get("source")
        return value if isinstance(value, str) else "unknown"

    def render(self) -> str:
        """Render as a labelled block for the prompt."""
        return (
            f"[{self.marker}] (company: {self.company}, source: {self.source}, "
            f"chunk: {self.chunk_id})\n{self.document.page_content}"
        )


def count_tokens(text: str) -> int:
    """Count tokens the way the model will."""
    return len(tiktoken.get_encoding(_ENCODING).encode(text))


def build_context(
    documents: Sequence[Document], *, max_tokens: int
) -> tuple[list[ContextBlock], str]:
    """Number the retrieved chunks and render them within a token budget.

    Args:
        documents: Retrieved chunks, best first.
        max_tokens: Ceiling for the rendered context.

    Returns:
        The blocks that fit, and the rendered context string.
    """
    blocks: list[ContextBlock] = []
    used = 0

    for position, document in enumerate(documents, start=1):
        block = ContextBlock(marker=position, document=document)
        cost = count_tokens(document.page_content) + _BLOCK_OVERHEAD_TOKENS

        if used + cost > max_tokens:
            # Ranked input: everything after this is weaker still, so stop
            # rather than skipping ahead and reordering the evidence.
            log.warning(
                "context budget reached, dropping lower-ranked chunks",
                extra={
                    "kept": len(blocks),
                    "dropped": len(documents) - len(blocks),
                    "max_tokens": max_tokens,
                },
            )
            break

        blocks.append(block)
        used += cost

    rendered = "\n\n".join(block.render() for block in blocks)
    log.debug("built context", extra={"blocks": len(blocks), "tokens": used})
    return blocks, rendered
