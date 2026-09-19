"""Logging setup for the agent audit trail (agents/trace.py).

Both entrypoints call `configure_logging()`. It exists because neither one
shows the audit lines by default: uvicorn's dictConfig configures only its
own loggers and leaves the root logger at WARNING with no handlers, so every
INFO-level audit record is dropped (an agent's ERROR still surfaces, which is
why a failing run logs but a successful one doesn't), and the CLI starts with
no logging configured at all.

The handler goes on the package logger rather than the root logger so
uvicorn's own access/error formatting is left untouched. Propagation stays
on, so pytest's `caplog` (a root handler) still captures agent records.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

PACKAGE_LOGGER_NAME = "equity_ensemble"
HANDLER_NAME = "equity_ensemble_audit"
DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(*, stream: TextIO | None = None) -> logging.Logger:
    """Send `equity_ensemble.*` audit records to `stream` (stderr by default).

    Level comes from `LOG_LEVEL`, defaulting to INFO. Idempotent: the named
    handler is only attached once, so calling this from both entrypoints, or
    twice in one process, will not duplicate every line.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(os.environ.get("LOG_LEVEL", DEFAULT_LEVEL).upper())

    if not any(handler.name == HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
