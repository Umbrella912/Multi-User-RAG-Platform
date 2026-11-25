"""Structured JSON logging configuration.

Configures the root logger (and common library loggers) to emit single-line
JSON records containing a timestamp, level, logger name, message, and any
extra structured fields passed via ``logger.info(..., extra={...})``.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes that belong to every LogRecord and must not be re-emitted as extras.
_RESERVED: set[str] = set(logging.makeLogRecord({}).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` instances as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` to a JSON string with structured context."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root and common library loggers to emit JSON to stdout.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "alembic"):
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.addHandler(handler)
        lib_logger.propagate = False
        lib_logger.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a logger bound to ``name`` (usually ``__name__``)."""
    return logging.getLogger(name)
