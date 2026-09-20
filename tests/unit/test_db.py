"""Unit coverage for FakeDatabase.get_recent_forecast in isolation — the
24h forecast cache's lookup logic, without the full graph/agent machinery
already exercising it indirectly in tests/integration/test_build_graph.py.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from equity_ensemble.persistence.db import FakeDatabase
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport


def _report(*, ticker: str, horizon_days: int, generated_at: datetime) -> ForecastReport:
    claim = AgentClaim(
        agent="technicals",
        ticker=ticker,
        direction="bullish",
        magnitude_bps=100,
        confidence=0.6,
        evidence=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
        falsifiers=["reversal"],
    )
    return ForecastReport(
        run_id=uuid4(),
        ticker=ticker,
        horizon_days=horizon_days,
        generated_at=generated_at,
        distribution=Distribution(bullish_pct=50, neutral_pct=30, bearish_pct=20),
        recommendation="bullish_lean",
        overall_confidence=0.6,
        escalate_to_analyst=False,
        thesis="Stub thesis.",
        agent_claims=[claim],
    )


class TestGetRecentForecast:
    async def test_no_forecasts_at_all_returns_none(self):
        db = FakeDatabase()
        assert await db.get_recent_forecast("AAPL", 21, max_age=timedelta(hours=24)) is None

    async def test_finds_a_forecast_within_max_age(self):
        db = FakeDatabase()
        report = _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        await db.save_forecast(report)

        found = await db.get_recent_forecast("AAPL", 21, max_age=timedelta(hours=24))

        assert found is not None
        assert found.run_id == report.run_id

    async def test_forecast_older_than_max_age_is_not_returned(self):
        db = FakeDatabase()
        old = datetime.now(UTC) - timedelta(hours=25)
        await db.save_forecast(_report(ticker="AAPL", horizon_days=21, generated_at=old))

        found = await db.get_recent_forecast("AAPL", 21, max_age=timedelta(hours=24))

        assert found is None

    async def test_boundary_just_inside_max_age_is_included(self):
        """`>=`, not `>` — a forecast right at the edge of the window still
        counts, matching the SQL (`generated_at >= $3`). Uses a one-second
        margin rather than the exact boundary: `get_recent_forecast` takes
        its own `datetime.now(UTC)` internally, a moment after the test
        takes its own — comparing against the razor's edge would be flaky
        by the microseconds between those two calls, not by the logic
        under test."""
        db = FakeDatabase()
        just_inside = datetime.now(UTC) - timedelta(hours=24) + timedelta(seconds=1)
        report = _report(ticker="AAPL", horizon_days=21, generated_at=just_inside)
        await db.save_forecast(report)

        found = await db.get_recent_forecast("AAPL", 21, max_age=timedelta(hours=24))

        assert found is not None
        assert found.run_id == report.run_id

    async def test_different_ticker_is_not_a_match(self):
        db = FakeDatabase()
        await db.save_forecast(
            _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        )

        assert await db.get_recent_forecast("MSFT", 21, max_age=timedelta(hours=24)) is None

    async def test_ticker_match_is_case_insensitive(self):
        db = FakeDatabase()
        report = _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        await db.save_forecast(report)

        found = await db.get_recent_forecast("aapl", 21, max_age=timedelta(hours=24))

        assert found is not None
        assert found.run_id == report.run_id

    async def test_different_horizon_days_is_not_a_match(self):
        db = FakeDatabase()
        await db.save_forecast(
            _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        )

        assert await db.get_recent_forecast("AAPL", 5, max_age=timedelta(hours=24)) is None

    async def test_returns_the_most_recent_of_multiple_matches(self):
        db = FakeDatabase()
        now = datetime.now(UTC)
        older = _report(ticker="AAPL", horizon_days=21, generated_at=now - timedelta(hours=10))
        newer = _report(ticker="AAPL", horizon_days=21, generated_at=now - timedelta(hours=1))
        await db.save_forecast(older)
        await db.save_forecast(newer)

        found = await db.get_recent_forecast("AAPL", 21, max_age=timedelta(hours=24))

        assert found is not None
        assert found.run_id == newer.run_id
