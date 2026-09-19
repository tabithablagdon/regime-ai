"""Track D "done when" check (PRD milestone M5 DoD): POST /forecast and the
CLI both produce identical Markdown/JSON output for the same ticker, when
wired to the same compiled graph."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from equity_ensemble.api.main import create_app
from equity_ensemble.cli.main import run_forecast_and_write
from equity_ensemble.data.fmp_client import RecordedFMPClient
from equity_ensemble.graph.build_graph import build_graph
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_CANNED_RUN_ID = uuid4()


def _canned_report() -> ForecastReport:
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
        run_id=_CANNED_RUN_ID,
        ticker="AAPL",
        horizon_days=21,
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
        distribution=Distribution(bullish_pct=55, neutral_pct=30, bearish_pct=15),
        recommendation="bullish_lean",
        overall_confidence=0.7,
        escalate_to_analyst=False,
        thesis="Deterministic stub thesis for the M5 parity check.",
        agent_claims=[claim],
    )


def _build_stub_graph():
    async def technicals(ticker, horizon_days):
        return _canned_report().agent_claims[0]

    async def fundamentals(ticker, horizon_days):
        return _canned_report().agent_claims[0].model_copy(
            update={"agent": "fundamentals_sentiment", "regime_label": None, "sentiment_score": 60}
        )

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        return _canned_report()

    return build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )


async def test_api_and_cli_produce_identical_output(tmp_path, monkeypatch):
    graph = _build_stub_graph()
    # get_profile is monkeypatched below, so no fixture file is needed for it.
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    # --- API ---
    app = create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL", "horizon_days": 21})
    assert response.status_code == 200
    api_body = response.json()
    api_markdown = api_body.pop("report_markdown")

    # --- CLI ---
    monkeypatch.chdir(tmp_path)
    cli_markdown = await run_forecast_and_write("AAPL", 21, graph=graph, fmp=fmp)

    md_path = tmp_path / f"forecast_AAPL_{_CANNED_RUN_ID}.md"
    json_path = tmp_path / f"forecast_AAPL_{_CANNED_RUN_ID}.json"
    assert md_path.exists()
    assert json_path.exists()

    cli_json_body = json.loads(json_path.read_text())

    assert cli_markdown == api_markdown
    assert cli_json_body["run_id"] == api_body["run_id"]
    assert cli_json_body["thesis"] == api_body["thesis"]
    assert cli_json_body["distribution"] == api_body["distribution"]


async def test_unknown_ticker_returns_404_from_api(monkeypatch):
    from equity_ensemble.data.fmp_client import TickerNotFoundError

    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "ZZZZ"})

    assert response.status_code == 404


async def test_engine_failure_returns_503_not_a_raw_500(monkeypatch):
    """When the reasoning pipeline itself fails (e.g. the LLM call isn't
    hooked up to a real key), the API must return a clean, distinguishable
    503 rather than leaking a stack trace as a generic 500."""
    from equity_ensemble.api.main import ENGINE_UNAVAILABLE_DETAIL
    from equity_ensemble.llm.client import LLMError

    async def technicals(ticker, horizon_days):
        return _canned_report().agent_claims[0]

    async def fundamentals(ticker, horizon_days):
        return _canned_report().agent_claims[0]

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        raise LLMError("simulated: OPENROUTER_API_KEY is not a real key")

    graph = build_graph(
        technicals_agent=technicals, fundamentals_agent=fundamentals, meta_agent=meta
    )
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL"})

    assert response.status_code == 503
    assert response.json()["detail"] == ENGINE_UNAVAILABLE_DETAIL


async def test_unknown_ticker_exits_nonzero_from_cli(tmp_path, monkeypatch):
    from equity_ensemble.data.fmp_client import TickerNotFoundError

    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)
    monkeypatch.chdir(tmp_path)

    import typer

    with pytest.raises(typer.BadParameter):
        await run_forecast_and_write("ZZZZ", 21, graph=graph, fmp=fmp)
