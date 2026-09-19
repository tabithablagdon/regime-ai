"""Track E "done when" check: the scoring job runs against at least one
aged forecast (PRD milestone M7 DoD)."""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from equity_ensemble.eval.brier_score import run_scoring_job
from equity_ensemble.persistence.db import FakeDatabase
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport


class _FakeFMP:
    def __init__(self, bars: list[dict]) -> None:
        self._bars = bars

    async def get_ohlcv(self, ticker: str, days: int = 504) -> list[dict]:
        return self._bars


def _make_report(
    *, generated_at: datetime, horizon_days: int, bullish_pct: float
) -> ForecastReport:
    claim = AgentClaim(
        agent="technicals",
        ticker="AAPL",
        direction="bullish",
        magnitude_bps=150,
        confidence=0.7,
        regime_label="Trending Bull",
        regime_changed=False,
        evidence=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
        falsifiers=["reversal"],
    )
    return ForecastReport(
        run_id=uuid4(),
        ticker="AAPL",
        horizon_days=horizon_days,
        generated_at=generated_at,
        distribution=Distribution(
            bullish_pct=bullish_pct,
            neutral_pct=(100 - bullish_pct) / 2,
            bearish_pct=(100 - bullish_pct) / 2,
        ),
        recommendation="bullish_lean",
        overall_confidence=0.7,
        escalate_to_analyst=False,
        thesis="Sample thesis.",
        agent_claims=[claim],
    )


async def test_scores_an_aged_forecast_and_writes_outcome_back():
    db = FakeDatabase()
    report = _make_report(generated_at=datetime(2026, 1, 1), horizon_days=5, bullish_pct=70)
    await db.save_forecast(report)

    fmp = _FakeFMP(
        [
            {"date": "2026-01-01", "close": 100.0},
            {"date": "2026-01-06", "close": 103.0},  # +300 bps -> realized bullish
        ]
    )

    scored = await run_scoring_job(db, fmp, as_of=date(2026, 1, 10))

    assert scored == [str(report.run_id)]
    outcome = db._outcomes[str(report.run_id)]
    assert outcome["realized_return_bps"] == 300.0
    assert outcome["brier_score"] < 0.33  # confident-and-correct beats the random baseline


async def test_forecast_not_yet_at_horizon_is_not_scored():
    db = FakeDatabase()
    report = _make_report(generated_at=datetime(2026, 1, 1), horizon_days=21, bullish_pct=70)
    await db.save_forecast(report)
    fmp = _FakeFMP([{"date": "2026-01-01", "close": 100.0}])

    scored = await run_scoring_job(db, fmp, as_of=date(2026, 1, 5))

    assert scored == []


async def test_already_scored_forecast_is_not_rescored():
    db = FakeDatabase()
    report = _make_report(generated_at=datetime(2026, 1, 1), horizon_days=5, bullish_pct=70)
    await db.save_forecast(report)
    await db.record_realized_outcome(
        str(report.run_id), realized_return_bps=100.0, brier_score=0.1
    )
    fmp = _FakeFMP([{"date": "2026-01-01", "close": 100.0}])

    scored = await run_scoring_job(db, fmp, as_of=date(2026, 1, 10))

    assert scored == []
