"""Locate and read raw filings from disk.

This module owns exactly one piece of knowledge: **the filename convention**.
`aapl-2025.htm` encodes the company and the fiscal year, and parsing it here
means no other module has to guess which filing it is looking at — the company
key that ends up in every chunk's metadata, and later in every Qdrant payload
filter, originates from this one regex.

The convention is enforced rather than inferred. A file named `AAPL_2025.html`
raises instead of silently producing an uppercase company key that will never
match a lowercase payload filter — a failure that is invisible until retrieval
returns zero results with no error at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# `aapl-2025.htm` → company "aapl", fiscal year 2025.
# Lowercase-only by design: company keys are lowercase everywhere in this
# system, so a capitalised filename is a convention violation, not a variant.
_FILENAME_PATTERN = re.compile(r"^(?P<company>[a-z][a-z0-9]{0,9})-(?P<year>\d{4})\.html?$")


@dataclass(frozen=True, slots=True)
class FilingSource:
    """A raw filing on disk, with its identity resolved from the filename."""

    path: Path
    company: str
    fiscal_year: int

    @property
    def source(self) -> str:
        """Filename, used as the `source` metadata value on every chunk."""
        return self.path.name


def parse_filing_name(path: Path) -> FilingSource:
    """Resolve a filing's company and fiscal year from its filename.

    Args:
        path: Path to a raw filing, named `{company}-{year}.htm`.

    Returns:
        The parsed `FilingSource`.

    Raises:
        IngestionError: If the filename does not follow the convention.
    """
    match = _FILENAME_PATTERN.match(path.name)
    if match is None:
        raise IngestionError(
            f"Filing name {path.name!r} does not match the required "
            f"'{{company}}-{{year}}.htm' convention (lowercase company, 4-digit year)."
        )
    return FilingSource(
        path=path,
        company=match["company"],
        fiscal_year=int(match["year"]),
    )


def discover_filings(raw_dir: Path) -> list[FilingSource]:
    """Find every filing in `raw_dir`, sorted by company for stable ordering.

    Stable ordering matters: chunk IDs are positional, so an unordered
    directory listing would produce different IDs on different machines and
    quietly break re-indexing idempotency.

    Args:
        raw_dir: Directory holding the raw filings.

    Returns:
        Parsed sources, ordered by company then fiscal year.

    Raises:
        IngestionError: If the directory is missing or holds no filings.
    """
    if not raw_dir.is_dir():
        raise IngestionError(f"Raw data directory not found: {raw_dir}")

    sources = [parse_filing_name(path) for path in sorted(raw_dir.glob("*.htm*"))]
    if not sources:
        raise IngestionError(f"No filings found in {raw_dir} (expected '{{company}}-{{year}}.htm')")

    sources.sort(key=lambda s: (s.company, s.fiscal_year))
    log.debug(
        "discovered filings",
        extra={"count": len(sources), "companies": [s.company for s in sources]},
    )
    return sources


def read_filing(source: FilingSource) -> str:
    """Read a filing's raw HTML.

    Decoding errors are replaced rather than raised: SEC filings occasionally
    carry stray bytes, and losing one character is preferable to failing an
    entire 8 MB document.

    Args:
        source: The filing to read.

    Returns:
        Raw HTML.

    Raises:
        IngestionError: If the file is missing or empty.
    """
    try:
        html = source.path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IngestionError(f"Could not read filing {source.path}: {exc}") from exc

    if not html.strip():
        raise IngestionError(f"Filing {source.path} is empty")

    log.debug("read filing", extra={"company": source.company, "bytes": len(html)})
    return html
