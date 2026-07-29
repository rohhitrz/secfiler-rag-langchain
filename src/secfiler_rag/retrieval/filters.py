"""Filter vocabulary shared by every retrieval strategy.

Dense search pushes filters down into Qdrant; sparse search applies them in
Python over an in-memory corpus. Same *meaning*, two mechanisms — so the
vocabulary lives in one place and each strategy translates it.

Without this module the two would drift: dense would accept a filter key that
sparse silently ignored, and a hybrid query would return one company's chunks
from one retriever and everyone's from the other. That failure produces
plausible wrong results, not an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qdrant_client.http import models as qmodels

from secfiler_rag.core.exceptions import RetrievalError
from secfiler_rag.indexing.collection import METADATA_PAYLOAD_KEY

# Filter keys every strategy must support, mapped to their Qdrant payload path.
# QdrantVectorStore nests document metadata, so the path is prefixed — a filter
# on the bare field name matches nothing and raises nothing.
FILTER_FIELDS: dict[str, str] = {"company": f"{METADATA_PAYLOAD_KEY}.company"}


def validate_filters(filters: Mapping[str, Any] | None) -> None:
    """Reject filter keys no strategy knows how to apply.

    Raises:
        RetrievalError: If a key is unsupported. Ignoring it instead would
            return unfiltered results that look like a quality problem rather
            than a typo.
    """
    if not filters:
        return
    unknown = sorted(set(filters) - set(FILTER_FIELDS))
    if unknown:
        raise RetrievalError(f"Unknown filter key(s) {unknown}. Supported: {sorted(FILTER_FIELDS)}")


def build_qdrant_filter(filters: Mapping[str, Any] | None) -> qmodels.Filter | None:
    """Translate filters into a Qdrant `Filter` for server-side matching."""
    validate_filters(filters)
    if not filters:
        return None

    conditions: list[qmodels.Condition] = [
        qmodels.FieldCondition(key=FILTER_FIELDS[key], match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return qmodels.Filter(must=conditions)


def matches(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Check a document's metadata against filters, for in-memory strategies.

    Compares against the *unprefixed* key, because an in-memory `Document`
    holds its metadata flat — the `metadata.` prefix is a Qdrant payload path,
    not part of the field name.
    """
    validate_filters(filters)
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())
