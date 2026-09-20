"""Track D "done when" check for the graph itself: both specialist nodes run
(and a failed one degrades to None rather than crashing the run), and the
synthesize node receives whatever the fan-out produced."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from equity_ensemble.agents.base import AgentUnavailable
from equity_ensemble.graph.build_graph import build_graph, run_forecast
from equity_ensemble.persistence.db import FakeDatabase
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport


def _claim(agent: str) -> AgentClaim:
    return AgentClaim(
        agent=agent,
        ticker="AAPL",
        direction="bullish",
        magnitude_bps=100,
        confidence=0.6,
        evidence=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
        falsifiers=["reversal"],
    )


def _report(**overrides) -> ForecastReport:
    claim = _claim("technicals")
    defaults = dict(
        run_id=uuid4(),
        ticker="AAPL",
        horizon_days=21,
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
        distribution=Distribution(bullish_pct=50, neutral_pct=30, bearish_pct=20),
        recommendation="bullish_lean",
        overall_confidence=0.6,
        escalate_to_analyst=False,
        thesis="Stub thesis.",
        agent_claims=[claim],
    )
    defaults.update(overrides)
    return ForecastReport(**defaults)


async def test_both_specialists_succeed_and_reach_synthesis():
    seen: dict[str, object] = {}

    async def technicals(ticker, horizon_days):
        return _claim("technicals")

    async def fundamentals(ticker, horizon_days):
        return _claim("fundamentals_sentiment")

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        seen["technicals"] = technicals_claim
        seen["fundamentals"] = fundamentals_claim
        return _report()

    graph = build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )
    result = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(result.report, ForecastReport)
    assert result.cache_hit is False
    assert seen["technicals"].agent == "technicals"
    assert seen["fundamentals"].agent == "fundamentals_sentiment"


async def test_one_agent_unavailable_degrades_to_none_not_a_crash():
    seen: dict[str, object] = {}

    async def technicals(ticker, horizon_days):
        raise AgentUnavailable("simulated failure")

    async def fundamentals(ticker, horizon_days):
        return _claim("fundamentals_sentiment")

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        seen["technicals"] = technicals_claim
        seen["fundamentals"] = fundamentals_claim
        return _report()

    graph = build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )
    result = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(result.report, ForecastReport)
    assert seen["technicals"] is None
    assert seen["fundamentals"].agent == "fundamentals_sentiment"


async def test_vendor_failure_degrades_to_single_agent_mode(caplog):
    """A rate limit or outage inside one specialist must cost only that
    specialist's claim, not the whole run (PRD §11 graceful degradation)."""
    seen: dict[str, object] = {}

    async def technicals(ticker, horizon_days):
        return _claim("technicals")

    async def fundamentals(ticker, horizon_days):
        raise RuntimeError("voyage rate limit: 3 RPM / 10K TPM")

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        seen["technicals"] = technicals_claim
        seen["fundamentals"] = fundamentals_claim
        return _report()

    graph = build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )
    caplog.set_level(logging.INFO)
    result = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(result.report, ForecastReport)
    assert seen["fundamentals"] is None
    assert seen["technicals"].agent == "technicals"
    assert "fundamentals_sentiment unavailable ticker=AAPL" in caplog.text
    assert "reason=RuntimeError" in caplog.text
    # The real cause is still recorded server-side, not swallowed.
    assert "voyage rate limit" in caplog.text


class _FakeCompiledGraph:
    """Stands in for a compiled LangGraph graph — just records what `config`
    it was invoked with, so config-forwarding can be tested without
    exercising real LangGraph callback machinery."""

    def __init__(self, report: ForecastReport) -> None:
        self._report = report
        self.received_config: object = "not called"

    async def ainvoke(self, state, config=None):
        self.received_config = config
        return {"report": self._report}


async def test_run_forecast_forwards_config_to_ainvoke():
    """config={"callbacks": [tracer]} (api/main.py's LangSmith wiring) must
    actually reach the graph invocation, not be silently dropped."""
    fake_graph = _FakeCompiledGraph(_report())
    sentinel_config = {"callbacks": ["sentinel-tracer"]}

    await run_forecast(fake_graph, ticker="AAPL", horizon_days=21, config=sentinel_config)

    assert fake_graph.received_config is sentinel_config


async def test_run_forecast_defaults_config_to_none():
    fake_graph = _FakeCompiledGraph(_report())

    await run_forecast(fake_graph, ticker="AAPL", horizon_days=21)

    assert fake_graph.received_config is None


class TestForecastCaching:
    """A repeat request for the same (ticker, horizon_days) within 24h must
    skip the whole pipeline — no LLM, FMP, or embedding calls — and a fresh
    run must persist so a later request can find it."""

    def _refusing_agents(self):
        """Agents that fail the test if the graph is ever actually invoked
        — proves a cache hit short-circuits before the graph runs at all,
        not just before the LLM call within it."""

        async def technicals(ticker, horizon_days):
            raise AssertionError("technicals ran on what should have been a cache hit")

        async def fundamentals(ticker, horizon_days):
            raise AssertionError("fundamentals ran on what should have been a cache hit")

        async def meta(ticker, horizon_days, t, f):
            raise AssertionError("meta_agent ran on what should have been a cache hit")

        return technicals, fundamentals, meta

    async def test_recent_forecast_is_served_from_cache_without_running_the_graph(self):
        db = FakeDatabase()
        cached = _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        await db.save_forecast(cached)

        technicals, fundamentals, meta = self._refusing_agents()
        graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )

        result = await run_forecast(graph, ticker="AAPL", horizon_days=21, db=db)

        assert result.cache_hit is True
        assert result.report.run_id == cached.run_id

    async def test_fresh_run_is_persisted_for_a_later_cache_hit(self):
        db = FakeDatabase()

        async def technicals(ticker, horizon_days):
            return _claim("technicals")

        async def fundamentals(ticker, horizon_days):
            return _claim("fundamentals_sentiment")

        async def meta(ticker, horizon_days, t, f):
            return _report(ticker=ticker, horizon_days=horizon_days, generated_at=datetime.now(UTC))

        real_graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )
        first = await run_forecast(real_graph, ticker="AAPL", horizon_days=21, db=db)
        assert first.cache_hit is False

        # A second call, wired to agents that fail if ever invoked — this
        # only passes if the fresh run above was actually persisted and the
        # cache check finds it, short-circuiting before the graph runs.
        refusing_technicals, refusing_fundamentals, refusing_meta = self._refusing_agents()
        refusing_graph = build_graph(
            technicals_agent=refusing_technicals,
            fundamentals_agent=refusing_fundamentals,
            meta_agent=refusing_meta,
        )
        second = await run_forecast(refusing_graph, ticker="AAPL", horizon_days=21, db=db)

        assert second.cache_hit is True
        assert second.report.run_id == first.report.run_id

    async def test_cached_report_older_than_max_age_is_not_used(self):
        db = FakeDatabase()
        stale = _report(
            ticker="AAPL",
            horizon_days=21,
            generated_at=datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        await db.save_forecast(stale)

        async def technicals(ticker, horizon_days):
            return _claim("technicals")

        async def fundamentals(ticker, horizon_days):
            return _claim("fundamentals_sentiment")

        fresh = _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))

        async def meta(ticker, horizon_days, t, f):
            return fresh

        graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )
        result = await run_forecast(graph, ticker="AAPL", horizon_days=21, db=db)

        assert result.cache_hit is False
        assert result.report.run_id == fresh.run_id

    async def test_cache_key_includes_horizon_days(self):
        db = FakeDatabase()
        await db.save_forecast(
            _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        )

        async def technicals(ticker, horizon_days):
            return _claim("technicals")

        async def fundamentals(ticker, horizon_days):
            return _claim("fundamentals_sentiment")

        fresh = _report(ticker="AAPL", horizon_days=5)

        async def meta(ticker, horizon_days, t, f):
            return fresh

        graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )
        # Different horizon than what's cached -> must not hit the cache.
        result = await run_forecast(graph, ticker="AAPL", horizon_days=5, db=db)

        assert result.cache_hit is False
        assert result.report.run_id == fresh.run_id

    async def test_cache_max_age_none_disables_the_cache_check(self):
        db = FakeDatabase()
        cached = _report(ticker="AAPL", horizon_days=21, generated_at=datetime.now(UTC))
        await db.save_forecast(cached)

        async def technicals(ticker, horizon_days):
            return _claim("technicals")

        async def fundamentals(ticker, horizon_days):
            return _claim("fundamentals_sentiment")

        fresh = _report(ticker="AAPL", horizon_days=21)

        async def meta(ticker, horizon_days, t, f):
            return fresh

        graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )
        result = await run_forecast(
            graph, ticker="AAPL", horizon_days=21, db=db, cache_max_age=None
        )

        assert result.cache_hit is False
        assert result.report.run_id == fresh.run_id

    async def test_no_db_means_no_caching_and_no_persistence(self):
        """Existing callers that never pass `db` (most of this file's other
        tests) must be completely unaffected — this is the regression
        guard for that."""

        async def technicals(ticker, horizon_days):
            return _claim("technicals")

        async def fundamentals(ticker, horizon_days):
            return _claim("fundamentals_sentiment")

        async def meta(ticker, horizon_days, t, f):
            return _report()

        graph = build_graph(
            technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
        )

        first = await run_forecast(graph, ticker="AAPL", horizon_days=21)
        second = await run_forecast(graph, ticker="AAPL", horizon_days=21)

        assert first.cache_hit is False
        assert second.cache_hit is False


async def test_both_agents_unavailable_propagates_meta_agents_error():
    async def technicals(ticker, horizon_days):
        raise AgentUnavailable("simulated failure")

    async def fundamentals(ticker, horizon_days):
        raise AgentUnavailable("simulated failure")

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        if technicals_claim is None and fundamentals_claim is None:
            raise ValueError("no claims available")
        return _report()

    graph = build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )

    with pytest.raises(ValueError):
        await run_forecast(graph, ticker="AAPL", horizon_days=21)
