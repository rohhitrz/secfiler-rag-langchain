"""Exception hierarchy for the project.

Why a hierarchy instead of raising `ValueError` everywhere: a caller (a CLI, an
API handler, a retry decorator) needs to distinguish "the user gave me a bad
filing path" from "OpenAI is rate-limiting me" from "a real bug". A single root
class also lets an outer boundary write `except SecfilerRagError` and be certain
it is catching *our* failures, never swallowing an unrelated library bug.
"""

from __future__ import annotations


class SecfilerRagError(Exception):
    """Base class for every error this project raises deliberately."""


class ConfigurationError(SecfilerRagError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised at startup or at first use of a credential — never mid-request, so
    that misconfiguration fails fast and loudly.
    """


class IngestionError(SecfilerRagError):
    """A source filing could not be loaded, cleaned, or split."""


class RetrievalError(SecfilerRagError):
    """A retrieval strategy could not run.

    Covers a missing collection and malformed filters — not "no results", which
    is a valid outcome rather than an error.
    """


class EvaluationError(SecfilerRagError):
    """An evaluation dataset is missing, malformed, or internally invalid."""


class IndexingError(SecfilerRagError):
    """Embedding or vector-store writing failed.

    Covers collection misconfiguration (a dimension mismatch between the
    embedding model and an existing collection) as well as upsert failures.
    """
