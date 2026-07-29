"""Logging contract: JSON shape, `extra=` propagation, idempotent setup."""

import json
import logging

import pytest

from secfiler_rag.core.logging import JsonFormatter, configure_logging, get_logger


@pytest.fixture
def record() -> logging.LogRecord:
    return logging.LogRecord(
        name="secfiler_rag.ingestion",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="indexed filing",
        args=(),
        exc_info=None,
    )


def test_json_formatter_emits_one_parseable_object(record):
    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "secfiler_rag.ingestion"
    assert payload["message"] == "indexed filing"
    assert payload["timestamp"].endswith("+00:00")


def test_extra_fields_are_promoted_to_top_level(record):
    record.company = "aapl"
    record.chunks = 199

    payload = json.loads(JsonFormatter().format(record))

    assert payload["company"] == "aapl"
    assert payload["chunks"] == 199


def test_exceptions_are_serialised(record):
    try:
        raise ValueError("qdrant unreachable")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "qdrant unreachable" in payload["exception"]


def test_configure_logging_does_not_stack_handlers():
    configure_logging(force=True)
    configure_logging(force=True)

    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_is_idempotent_without_force():
    configure_logging(force=True, level="DEBUG")
    configure_logging(level="ERROR")  # should be a no-op

    assert logging.getLogger("secfiler_rag").level == logging.DEBUG


def test_third_party_loggers_stay_quiet_by_default():
    configure_logging(force=True, level="DEBUG")

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("secfiler_rag.retrieval").getEffectiveLevel() == logging.DEBUG


def test_get_logger_inherits_project_hierarchy():
    assert get_logger("secfiler_rag.indexing").name == "secfiler_rag.indexing"
