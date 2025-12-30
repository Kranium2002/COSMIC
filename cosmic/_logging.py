"""Structured logging helpers for reproducible experiments."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_DEFAULT_LEVEL = "INFO"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        fields = record.__dict__.get("fields")
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str = "cosmic") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("COSMIC_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured log line with stable key ordering."""
    logger.info(event, extra={"fields": fields})
