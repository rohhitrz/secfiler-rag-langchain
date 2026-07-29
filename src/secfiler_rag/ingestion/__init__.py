"""Stage 1 — raw filing bytes to clean, metadata-carrying `Document` objects.

Responsibility: load SEC 10-K HTML from disk, strip non-content markup
(scripts, styles, inline-XBRL), and split the result into chunks that carry the
metadata retrieval will later filter on (`company`, `chunk_id`, `source`).

Output contract: `list[langchain_core.documents.Document]`. Every downstream
stage speaks `Document`, so swapping the loader or the splitter never ripples
past this package.

Status: not implemented yet (next module).
"""
