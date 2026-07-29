"""Centralised logging setup.

Why the standard library instead of `structlog` / `loguru`:

* Every dependency we use (LangChain, httpx, qdrant-client, openai) logs
  through `logging`. Configuring the stdlib root logger means we capture
  *their* diagnostics too — with a third-party logger we would own our lines
  and lose theirs, which is exactly the output you need when a retrieval call
  hangs.
* Zero extra dependency for a solved problem.

We still get structured output: `JsonFormatter` emits one JSON object per line
for log aggregators, while `console` mode stays readable during development.
Contextual fields travel via the standard `extra=` argument:

    log.info("indexed filing", extra={"company": "aapl", "chunks": 199})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Attributes present on every LogRecord. Anything *not* in this set was passed
# by the caller through `extra=`, and is therefore worth emitting as a field.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_configured = False


class JsonFormatter(logging.Formatter):
    """Render a `LogRecord` as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, promoting `extra=` keys to top-level fields."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "console",
    force: bool = False,
) -> None:
    """Install the project's log handler on the root logger.

    Idempotent by design: importing a module must never reconfigure logging
    behind the application's back, and repeated calls must not stack duplicate
    handlers (the classic cause of every line printing three times).

    Args:
        level: Root level for the `secfiler_rag` logger tree.
        fmt: `"console"` for human-readable output, `"json"` for aggregators.
        force: Reconfigure even if setup already ran. Intended for tests.
    """
    global _configured
    if _configured and not force:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_CONSOLE_FORMAT))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)

    # Our code logs at the configured level; noisy third parties stay at
    # WARNING unless we are explicitly debugging.
    root.setLevel(logging.WARNING)
    logging.getLogger("secfiler_rag").setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`.

    Always call with `__name__` so the logger inherits the `secfiler_rag.*`
    hierarchy and therefore the level set by `configure_logging`.
    """
    return logging.getLogger(name)
