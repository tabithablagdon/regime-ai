"""Renders a ForecastReport to the Markdown structure in PRD §10.3 — owned
by Track D. Pure function, no I/O, fully unit-testable / snapshot-testable
against a hand-built ForecastReport fixture.

Structure (§10.3):
  1. Header: ticker, horizon, generated timestamp, recommendation, confidence
  2. Probability distribution table (bullish/neutral/bearish %)
  3. Thesis paragraph
  4. Dissent log: both agents' raw claims side by side, + critique exchange if any
  5. Citations: every filing/news source referenced, with date
  6. Footer disclaimer, printed unconditionally on every report
"""

from __future__ import annotations

from equity_ensemble.schemas.models import ForecastReport

DISCLAIMER_FOOTER = (
    "Decision support only — not a trade recommendation. "
    "Requires human review before any action."
)


def render_markdown(report: ForecastReport) -> str:
    raise NotImplementedError("Track D: implement per the §10.3 structure above")
