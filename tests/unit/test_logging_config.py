"""Regression tests for the audit trail actually reaching a log stream.

The bug these cover: under uvicorn's logging config the root logger sits at
WARNING with no handlers, so every INFO-level agent audit line was dropped
and a successful forecast left no trace in `.run/api.log` (only failures,
which log at ERROR, showed up).
"""

from __future__ import annotations

import io
import logging
import logging.config

import pytest
from uvicorn.config import LOGGING_CONFIG

from equity_ensemble.agents.trace import log_event
from equity_ensemble.logging_config import (
    HANDLER_NAME,
    PACKAGE_LOGGER_NAME,
    configure_logging,
)


@pytest.fixture
def clean_package_logger():
    """Restores the package logger, since these tests mutate global state."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    original_handlers, original_level = list(logger.handlers), logger.level
    yield logger
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def test_agent_info_lines_are_dropped_under_uvicorn_defaults_alone(clean_package_logger):
    """Uvicorn configures only its own loggers, so ours inherit the root
    level of WARNING and discard INFO before any handler is consulted."""
    logging.config.dictConfig(LOGGING_CONFIG)
    agent_logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals")

    assert agent_logger.getEffectiveLevel() == logging.WARNING
    assert not agent_logger.isEnabledFor(logging.INFO)
    assert agent_logger.isEnabledFor(logging.ERROR)  # which is why failures did log


def test_configure_logging_makes_agent_info_lines_visible(clean_package_logger):
    logging.config.dictConfig(LOGGING_CONFIG)
    stream = io.StringIO()

    configure_logging(stream=stream)
    log_event(
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals"),
        "technicals",
        "decision",
        ticker="AAPL",
        direction="bullish",
    )

    assert "technicals decision ticker=AAPL direction=bullish" in stream.getvalue()


def test_configure_logging_is_idempotent(clean_package_logger):
    stream = io.StringIO()

    configure_logging(stream=stream)
    configure_logging(stream=stream)

    audit_handlers = [h for h in clean_package_logger.handlers if h.name == HANDLER_NAME]
    assert len(audit_handlers) == 1

    log_event(
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals"),
        "technicals",
        "called",
        ticker="AAPL",
    )
    assert stream.getvalue().count("technicals called") == 1


def test_log_level_env_var_is_respected(clean_package_logger, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    stream = io.StringIO()

    configure_logging(stream=stream)
    log_event(
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals"),
        "technicals",
        "called",
        ticker="AAPL",
    )

    assert stream.getvalue() == ""


async def test_api_lifespan_configures_logging_before_serving(clean_package_logger, monkeypatch):
    """The API's fix lives in the lifespan, which runs after uvicorn installs
    its own config. Without this the audit trail is CLI-only."""
    import equity_ensemble.graph.wiring as wiring
    from equity_ensemble.api.main import _production_lifespan, create_app_from_env

    class _StubDatabase:
        async def close(self) -> None: ...

    async def fake_build_production_graph():
        return object(), _StubDatabase(), object()

    monkeypatch.setattr(wiring, "build_production_graph", fake_build_production_graph)
    logging.config.dictConfig(LOGGING_CONFIG)
    agent_logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals")
    assert not agent_logger.isEnabledFor(logging.INFO)

    app = create_app_from_env()
    async with _production_lifespan(app):
        assert agent_logger.isEnabledFor(logging.INFO)
        assert any(h.name == HANDLER_NAME for h in clean_package_logger.handlers)


def test_records_still_propagate_so_caplog_keeps_working(clean_package_logger, caplog):
    configure_logging(stream=io.StringIO())
    caplog.set_level(logging.INFO)

    log_event(
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.agents.technicals"),
        "technicals",
        "called",
        ticker="AAPL",
    )

    assert "technicals called ticker=AAPL" in caplog.text
