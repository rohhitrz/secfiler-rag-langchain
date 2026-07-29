"""Turn SEC filing HTML into retrievable plain text.

This is the highest-leverage module in the pipeline. Everything downstream —
embeddings, BM25 tokens, the context the LLM finally reads — is built from
whatever text comes out of here. A number deleted at this stage cannot be
recovered by a better retriever, a bigger model, or a smarter prompt.

Two decisions define it:

**1. Inline XBRL is unwrapped, not removed.**
SEC filings are inline-XBRL documents: machine-readable tags wrap the *visible*
values. `<ix:nonFraction>416,161</ix:nonFraction>` is not metadata sitting
beside the revenue figure — it *is* the revenue figure. Deleting those tags
with their contents (the obvious reading of "strip the XBRL namespace") deletes
the financial data.

Measured on the Apple FY2025 filing: removing them outright dropped 251 spans of
real content, including `10-K`, `Apple Inc.`, the fiscal year end date, and 53%
of every digit in the document. So `ix:` tags are *unwrapped* — the tag goes,
the text stays — with the exception of the four container tags
(`ix:header`, `ix:hidden`, `ix:references`, `ix:resources`) whose contents
genuinely are machine-only: GAAP taxonomy URLs and period markers that would
otherwise pollute every embedding.

**2. Tables become rows, not word soup.**
A 10-K is mostly tables. Naive text extraction flattens an income statement into
a stream of labels followed by a stream of digits, so the label and its figure
end up hundreds of characters apart — retrievable, but unanswerable. Here each
`<tr>` becomes one line with cells joined by ` | `, keeping a row's label
adjacent to its numbers:

    Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285

This is what fixes the `Products $ $ $` failure carried over from the previous
build.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# Tags whose text is never human-readable content.
_JUNK_TAGS = ("script", "style", "head", "noscript")

# Inline-XBRL containers holding machine-only metadata — taxonomy URLs, context
# periods, hidden facts. Removed with their contents.
_XBRL_METADATA_TAGS = re.compile(r"^ix:(header|hidden|references|resources)$")

# Any inline-XBRL tag. Those not matched above wrap visible values and are
# unwrapped so the value survives.
_XBRL_ANY_TAG = re.compile(r"^ix:")

# A cell holding only a currency symbol belongs to the number that follows it;
# one holding only a percent or closing paren belongs to the number before it.
_LEADING_SYMBOLS = frozenset({"$"})
_TRAILING_SYMBOLS = frozenset({"%", ")"})

_INTRA_LINE_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{2,}")


def clean_html(html: str) -> str:
    """Extract retrievable plain text from a filing's raw HTML.

    Args:
        html: Raw filing HTML.

    Returns:
        Plain text: one logical block per line, table rows as ` | `-joined
        cells, with all inline-XBRL values preserved.

    Raises:
        IngestionError: If cleaning yields no text at all, which means the
            input was not a filing.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_JUNK_TAGS):
        tag.decompose()
    for tag in soup.find_all(_XBRL_METADATA_TAGS):
        tag.decompose()
    for tag in soup.find_all(_XBRL_ANY_TAG):
        tag.unwrap()

    _flatten_tables(soup)

    text = _normalise(soup.get_text(separator="\n"))
    if not text:
        raise IngestionError("Cleaning produced no text — input is not a filing document")
    return text


def _flatten_tables(soup: BeautifulSoup) -> None:
    """Replace each `<table>` with line-per-row, ` | `-separated text.

    Applied innermost-first so a nested table is already flattened text by the
    time its parent is processed. SEC filings do not currently nest tables, but
    the ordering costs nothing and removes a whole class of silent corruption.
    """
    tables = soup.find_all("table")
    for table in reversed(tables):
        rows = [
            " | ".join(cells)
            for row in table.find_all("tr")
            if (cells := _merge_symbol_cells(_row_cells(row)))
        ]
        # Wrapping newlines keep the table from fusing onto adjacent prose.
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))


def _row_cells(row: Tag) -> list[str]:
    """Extract non-empty cell texts from one table row.

    Empty cells are dropped: filings use them heavily for visual spacing, and
    keeping them produces rows like `| | | 416,161 | |` that add tokens and no
    meaning.
    """
    cells = []
    for cell in row.find_all(["td", "th"]):
        text = " ".join(cell.get_text(" ", strip=True).split())
        if text:
            cells.append(text)
    return cells


def _merge_symbol_cells(cells: Iterable[str]) -> list[str]:
    """Reattach orphaned currency and percent symbols to their numbers.

    Filings put `$` and `%` in their own table cells for alignment, which
    extracts as `$ | 416,161 | 6 | %`. Rejoining them yields
    `$416,161 | 6%` — fewer tokens, and the value reads as one unit to both
    the embedding model and a human debugging a chunk.
    """
    merged: list[str] = []
    pending = ""
    for cell in cells:
        if cell in _LEADING_SYMBOLS:
            pending = cell
            continue
        if cell in _TRAILING_SYMBOLS and merged:
            merged[-1] += cell
            continue
        merged.append(pending + cell)
        pending = ""
    if pending:
        merged.append(pending)
    return merged


def _normalise(text: str) -> str:
    """Collapse whitespace while preserving line structure.

    The distinction matters. The previous build collapsed *all* whitespace
    including newlines, which destroyed every block boundary — and a recursive
    splitter with no paragraph or line boundaries to split on degenerates into
    blind character slicing. Runs of spaces within a line are collapsed; the
    newlines that mark structure are kept.
    """
    lines = (_INTRA_LINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n", "\n".join(line for line in lines if line)).strip()
