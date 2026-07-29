"""Split cleaned filing text into metadata-carrying `Document` chunks.

**What LangChain is doing here, and what it is not.**

`RecursiveCharacterTextSplitter` is one of the few places where the framework
earns its keep outright. It tries a list of separators in order — paragraph,
then line, then sentence, then word — and only falls back to a blind character
cut when nothing else fits. The previous build sliced every `chunk_size`
characters unconditionally, which cuts sentences (and table rows) in half.

What LangChain is *not* deciding: the chunk size, the overlap, the separator
list, or the metadata contract. Those are retrieval design decisions, and they
live here.

**Why overlap exists.** A fact split across a boundary is unretrievable by
either chunk — neither half contains the whole statement. Overlap means the
boundary region appears in both neighbours, so the fact survives in at least
one. The cost is index size: 200 of every 1000 characters are stored twice, so
~20% more chunks for the same corpus.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# Tried in order. Paragraph and line breaks come first because the cleaner
# preserves them deliberately — a table row is one line, so splitting on "\n"
# keeps rows intact rather than cutting through a set of figures.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_filing(
    text: str,
    *,
    company: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split cleaned text into chunks carrying the metadata retrieval needs.

    Args:
        text: Cleaned filing text.
        company: Lowercase ticker, stamped onto every chunk.
        source: Filename the text came from.
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters shared between neighbouring chunks.

    Returns:
        Documents with `company`, `chunk_id`, `source` and `start_index`
        metadata.

    Raises:
        IngestionError: If the text is empty or the split produces nothing.
    """
    if not text.strip():
        raise IngestionError(f"Cannot split empty text for company {company!r}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        # Records each chunk's offset in the source text. Cheap to keep, and
        # the only way to answer "where in the filing did this come from?"
        # when auditing a retrieval result.
        add_start_index=True,
    )

    chunks = splitter.split_text(text)
    if not chunks:
        raise IngestionError(f"Splitting produced no chunks for company {company!r}")

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "company": company,
                "chunk_id": chunk_id,
                "source": source,
                "start_index": start_index,
            },
        )
        for chunk_id, (chunk, start_index) in enumerate(
            zip(chunks, _start_indices(text, chunks), strict=True)
        )
    ]

    log.info(
        "split filing into chunks",
        extra={"company": company, "source": source, "chunks": len(documents)},
    )
    return documents


def _start_indices(text: str, chunks: list[str]) -> list[int]:
    """Recover each chunk's character offset in the source text.

    `split_text` returns bare strings, dropping the `start_index` that
    `create_documents` would have attached — so it is recomputed here with a
    forward-only scan. Forward-only matters: searching from position zero would
    match an earlier identical chunk (boilerplate repeats in filings) and
    report an offset that is off by thousands of characters.
    """
    indices = []
    cursor = 0
    for chunk in chunks:
        found = text.find(chunk, cursor)
        if found == -1:  # pragma: no cover — only reachable if the splitter mutates text
            found = cursor
        indices.append(found)
        cursor = found + 1
    return indices
