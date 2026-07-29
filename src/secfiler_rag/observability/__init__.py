"""Cross-cutting — tracing and run metadata (LangSmith).

Responsibility: wire LangSmith tracing on/off from configuration, tag runs with
the experiment metadata that makes A/B comparisons readable, and keep tracing
strictly optional so the pipeline still runs with no LangSmith credentials.

Status: not implemented yet.
"""
