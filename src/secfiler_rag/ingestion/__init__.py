"""Stage 1 — raw filing bytes to clean, metadata-carrying `Document` objects.

Responsibility: load SEC 10-K HTML from disk, strip non-content markup while
preserving inline-XBRL *values*, flatten tables into readable rows, and split
the result into chunks that carry the metadata retrieval will later filter on.

Output contract — every `Document` carries:

| Key | Type | Meaning |
|---|---|---|
| `company` | `str` | Lowercase ticker (`aapl`) |
| `chunk_id` | `int` | Position within that company's chunk list |
| `source` | `str` | Source filename |
| `start_index` | `int` | Character offset in the cleaned text |

Downstream stages speak `Document`, so replacing the cleaner or the splitter
never ripples past this package.
"""

from secfiler_rag.ingestion.cleaner import clean_html
from secfiler_rag.ingestion.loader import (
    FilingSource,
    discover_filings,
    parse_filing_name,
    read_filing,
)
from secfiler_rag.ingestion.pipeline import ingest_all, ingest_filing
from secfiler_rag.ingestion.splitter import split_filing

__all__ = [
    "FilingSource",
    "clean_html",
    "discover_filings",
    "ingest_all",
    "ingest_filing",
    "parse_filing_name",
    "read_filing",
    "split_filing",
]
