"""Load evaluation datasets from disk.

The loader is the only component that understands the dataset's *schema*. It
translates the on-disk shape into `EvalItem`s whose `filters` are an opaque
mapping — so the harness downstream can forward them to a retriever without
ever learning that a thing called "company" exists.

That translation is what keeps the harness domain-free. Put schema knowledge in
the harness and every future dataset change becomes a harness change, and the
harness stops being a fair judge of strategies it was not written for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from secfiler_rag.core.exceptions import EvaluationError
from secfiler_rag.core.logging import get_logger

log = get_logger(__name__)

# Keys in an item that are retrieval filters rather than eval bookkeeping.
_FILTER_KEYS = ("company",)


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One question and the evidence that proves it was answered.

    Attributes:
        query: The question, phrased as a user would ask it.
        expected_substring: Text that must appear in a retrieved chunk for the
            item to count as a hit.
        filters: Opaque retrieval constraints, forwarded verbatim to the
            retriever. The harness never inspects these.
        tier: 1 for lexical smoke tests, 2 for realistic questions.
    """

    query: str
    expected_substring: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    tier: int = 2


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A named, versioned set of evaluation items."""

    name: str
    version: int
    items: tuple[EvalItem, ...]

    def __len__(self) -> int:
        """Number of items in the dataset."""
        return len(self.items)

    def tier(self, tier: int) -> tuple[EvalItem, ...]:
        """Items belonging to one tier."""
        return tuple(item for item in self.items if item.tier == tier)


def load_dataset(path: Path) -> EvalDataset:
    """Read an evaluation dataset from JSON.

    Args:
        path: Path to the dataset file.

    Returns:
        The parsed dataset.

    Raises:
        EvaluationError: If the file is missing, malformed, empty, or an item
            lacks a query or expected substring.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"Evaluation dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Evaluation dataset {path} is not valid JSON: {exc}") from exc

    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise EvaluationError(f"Evaluation dataset {path} contains no items")

    items = tuple(_parse_item(entry, path, position) for position, entry in enumerate(raw_items))

    dataset = EvalDataset(
        name=str(raw.get("name", path.stem)),
        version=int(raw.get("version", 1)),
        items=items,
    )
    log.debug(
        "loaded eval dataset",
        extra={"dataset": dataset.name, "version": dataset.version, "items": len(dataset)},
    )
    return dataset


def _parse_item(entry: Any, path: Path, position: int) -> EvalItem:
    """Convert one raw entry, failing loudly rather than skipping it.

    A silently dropped item makes the denominator wrong, which is worse than a
    crash: the score still looks plausible.
    """
    if not isinstance(entry, dict):
        raise EvaluationError(f"{path} item {position} is not an object")

    query = entry.get("query")
    expected = entry.get("expected_substring")
    if not isinstance(query, str) or not query.strip():
        raise EvaluationError(f"{path} item {position} has no query")
    if not isinstance(expected, str) or not expected.strip():
        raise EvaluationError(f"{path} item {position} ({query!r}) has no expected_substring")

    filters = {key: entry[key] for key in _FILTER_KEYS if entry.get(key) is not None}
    return EvalItem(
        query=query,
        expected_substring=expected,
        filters=filters,
        tier=int(entry.get("tier", 2)),
    )
