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
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from equity_ensemble.data.fmp_client import FMPClient, TickerNotFoundError
from equity_ensemble.graph.build_graph import run_forecast
from equity_ensemble.render.markdown_report import render_markdown

logger = logging.getLogger(__name__)

ENGINE_UNAVAILABLE_DETAIL = "The analysis engine is not available right now."


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

        try:
            report = await run_forecast(
                app.state.graph, ticker=ticker, horizon_days=request.horizon_days
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

        return {**report.model_dump(mode="json"), "report_markdown": render_markdown(report)}


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
