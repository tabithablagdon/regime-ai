"""Typer CLI — Track D (PRD §10.2).

  forecast run AAPL --horizon 21

Writes forecast_AAPL_<run_id>.md and forecast_AAPL_<run_id>.json to the
working directory and prints the Markdown report to stdout.
`run_forecast_and_write` is the testable core — it calls the same
`graph.build_graph.run_forecast` the API calls, so behavior never diverges
between the two output surfaces (PRD's own stated goal for this split); the
Typer `run` command just wires production dependencies around it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from equity_ensemble.data.fmp_client import FMPClient, TickerNotFoundError
from equity_ensemble.graph.build_graph import run_forecast
from equity_ensemble.logging_config import configure_logging
from equity_ensemble.persistence.db import Database
from equity_ensemble.render.markdown_report import render_markdown

app = typer.Typer()


@app.callback()
def _cli() -> None:
    """Keeps `run` addressed as an explicit subcommand (`forecast run AAPL`,
    PRD §10.2) instead of Typer's single-command shortcut collapsing it."""
    configure_logging()


async def run_forecast_and_write(
    ticker: str, horizon: int, *, graph: Any, fmp: FMPClient, db: Database | None = None
) -> str:
    ticker = ticker.upper()
    try:
        await fmp.get_profile(ticker)
    except TickerNotFoundError as exc:
        raise typer.BadParameter(f"unknown or unsupported ticker: {ticker}") from exc

    result = await run_forecast(graph, ticker=ticker, horizon_days=horizon, db=db)
    report = result.report
    if result.cache_hit:
        typer.echo(
            f"Using a cached report generated {report.generated_at.isoformat()} "
            "(< 24h old) — skipping the agent pipeline.",
            err=True,
        )
    markdown = render_markdown(report)

    Path(f"forecast_{ticker}_{report.run_id}.md").write_text(markdown)
    Path(f"forecast_{ticker}_{report.run_id}.json").write_text(report.model_dump_json(indent=2))
    return markdown


async def _run_with_production_deps(ticker: str, horizon: int) -> None:
    from equity_ensemble.graph.wiring import build_production_graph

    graph, db, fmp = await build_production_graph()
    try:
        markdown = await run_forecast_and_write(ticker, horizon, graph=graph, fmp=fmp, db=db)
        typer.echo(markdown)
    finally:
        await db.close()


@app.command()
def run(ticker: str, horizon: int = 21) -> None:
    asyncio.run(_run_with_production_deps(ticker, horizon))


if __name__ == "__main__":
    app()
