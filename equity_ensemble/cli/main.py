"""Typer CLI — owned by Track D (PRD §10.2).

  forecast run AAPL --horizon 21

Writes forecast_AAPL_<run_id>.md and forecast_AAPL_<run_id>.json to the
working directory and prints the Markdown report to stdout. Must call the
same graph the API calls (equity_ensemble.graph.build_graph) so behavior
never diverges between the two output surfaces (PRD's own stated goal for
this CLI/API split).
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def run(ticker: str, horizon: int = 21) -> None:
    raise NotImplementedError(
        "Track D: run the compiled graph for `ticker`, write forecast_<TICKER>_<run_id>.{md,json}, "
        "print the Markdown report to stdout"
    )


if __name__ == "__main__":
    app()
