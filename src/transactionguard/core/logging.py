"""Structured logging with structlog.

Structured logs are key/value events (``event="prediction_scored" latency_ms=4.1``) instead of
free text, so they can be searched and aggregated. In containers we emit JSON lines.

Privacy rule: never log raw transaction payloads. Log identifiers, counts and timings only.
``redact`` strips known sensitive keys as a safety net.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

SENSITIVE_KEYS = {
    "password",
    "token",
    "access_token",
    "authorization",
    "transaction",
    "features",
    "payload",
    "jwt_secret",
}


def redact(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS:
            event_dict[key] = "[redacted]"
    return event_dict


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,  # e.g. request_id bound per request
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    # Route stdlib logging (uvicorn, celery, alembic) through the same level.
    logging.basicConfig(level=level.upper(), stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("mlflow", "alembic.runtime.migration", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
