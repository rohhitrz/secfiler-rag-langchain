"""secfiler-rag — an Advanced RAG system over SEC 10-K filings.

The package is organised as a one-way pipeline. Each sub-package owns exactly
one stage and depends only on the stages to its left plus `config`/`core`:

    ingestion -> indexing -> retrieval -> generation
                                  ^
                        evaluation | observability  (cross-cutting)

See `docs/03-folder-structure.md` for the full rationale.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
