"""Track D "done when" check for the graph itself: both specialist nodes run
(and a failed one degrades to None rather than crashing the run), and the
synthesize node receives whatever the fan-out produced."""

from __future__ import annotations

import logging
from datetime import date, datetime
from uuid import uuid4

import pytest

from equity_ensemble.agents.base import AgentUnavailable
from equity_ensemble.graph.build_graph import build_graph, run_forecast
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


def _report() -> ForecastReport:
    claim = _claim("technicals")
    return ForecastReport(
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
    report = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(report, ForecastReport)
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
    report = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(report, ForecastReport)
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
    report = await run_forecast(graph, ticker="AAPL", horizon_days=21)

    assert isinstance(report, ForecastReport)
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
