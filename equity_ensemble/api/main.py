"""FastAPI app — Track D (PRD §10.1).

POST /forecast  { "ticker": "AAPL", "horizon_days": 21 }
  -> 200 { ...ForecastReport fields..., "report_markdown": "..." }

Ticker validation (existence via FMP profile lookup) happens before the
graph fans out (PRD §4.2 step 2) — reject unknown tickers immediately.

`create_app` takes pre-built dependencies directly and registers no
lifespan, so tests can hand it stub/fake graphs and FMP clients without any
of the real OpenRouter/Postgres/Voyage machinery. `create_app_from_env` is
the production entrypoint (`uvicorn equity_ensemble.api.main:app`), which
wires everything for real via graph.wiring.build_production_graph inside a
lifespan hook.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from equity_ensemble.data.fmp_client import FMPClient, TickerNotFoundError
from equity_ensemble.graph.build_graph import run_forecast
from equity_ensemble.logging_config import configure_logging
from equity_ensemble.render.markdown_report import render_markdown

logger = logging.getLogger(__name__)

ENGINE_UNAVAILABLE_DETAIL = "The analysis engine is not available right now."


def _build_langsmith_tracer() -> Any | None:
    """Attaches LangSmith tracing to a single request rather than relying on
    the (also-supported) global `LANGSMITH_TRACING=true` env var, so we can
    grab that request's own trace URL and hand it back to the caller —
    that's what the web UI's "View execution trace" link points at. Purely
    opt-in: with no API key set (either name below), this returns None and
    nothing about a request changes.

    Checks both `LANGSMITH_*` and the legacy `LANGCHAIN_*` env var names —
    the underlying `langsmith` client already accepts either (it tries
    `LANGSMITH_` first, falls back to `LANGCHAIN_`); this gate needs to
    agree with that or it can refuse to trace when the client would have
    happily authenticated.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return None
    from langchain_core.tracers.langchain import LangChainTracer

    project = (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "equity-ensemble"
    )
    return LangChainTracer(project_name=project)


def _get_trace_url(tracer: Any | None) -> str | None:
    if tracer is None:
        return None
    try:
        return tracer.get_run_url()
    except Exception:
        # Tracing is a debugging aid, never load-bearing — a LangSmith
        # hiccup (bad key, project not yet created, network blip) must
        # never take down a request that otherwise succeeded.
        logger.warning("could not retrieve LangSmith trace URL", exc_info=True)
        return None


class ForecastRequest(BaseModel):
    ticker: str
    horizon_days: int = 21


def _register_routes(app: FastAPI) -> None:
    @app.post("/forecast")
    async def forecast(request: ForecastRequest) -> dict:
        ticker = request.ticker.upper()
        try:
            await app.state.fmp.get_profile(ticker)
        except TickerNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown or unsupported ticker: {ticker}"
            ) from exc
        except Exception:
            # A genuinely unknown ticker is TickerNotFoundError (above, 404).
            # Anything else here — FMP key not configured, FMP unreachable —
            # means the engine can't even validate the ticker, same class of
            # problem as the pipeline failure below.
            logger.exception("ticker validation failed for ticker=%s", ticker)
            raise HTTPException(status_code=503, detail=ENGINE_UNAVAILABLE_DETAIL) from None

        tracer = _build_langsmith_tracer()
        config = {"callbacks": [tracer]} if tracer else None
        try:
            report = await run_forecast(
                app.state.graph,
                ticker=ticker,
                horizon_days=request.horizon_days,
                config=config,
            )
        except Exception:
            # Any failure inside the agent/graph pipeline — an LLM call that
            # isn't hooked up to a real key, both specialists unavailable,
            # a vendor outage — is the same problem from the caller's point
            # of view: the reasoning engine didn't produce a report. Log the
            # real cause server-side but never leak a stack trace to the
            # client.
            logger.exception("forecast pipeline failed for ticker=%s", ticker)
            raise HTTPException(status_code=503, detail=ENGINE_UNAVAILABLE_DETAIL) from None

        return {
            **report.model_dump(mode="json"),
            "report_markdown": render_markdown(report),
            "trace_url": _get_trace_url(tracer),
        }


def create_app(*, graph: Any, fmp: FMPClient) -> FastAPI:
    """Testable factory: pass pre-built (possibly stub) dependencies."""
    app = FastAPI(title="Equity Forecast Ensemble", version="0.1.0")
    app.state.graph = graph
    app.state.fmp = fmp
    _register_routes(app)
    return app


@asynccontextmanager
async def _production_lifespan(app: FastAPI):
    from equity_ensemble.graph.wiring import build_production_graph

    # Runs after uvicorn has installed its own logging config, which would
    # otherwise swallow every INFO-level agent audit line.
    configure_logging()

    graph, db, fmp = await build_production_graph()
    app.state.graph = graph
    app.state.fmp = fmp
    try:
        yield
    finally:
        await db.close()


def create_app_from_env() -> FastAPI:
    """Production entrypoint — builds real dependencies (PRD §5's stack)."""
    app = FastAPI(
        title="Equity Forecast Ensemble", version="0.1.0", lifespan=_production_lifespan
    )
    _register_routes(app)
    return app


app = create_app_from_env()
