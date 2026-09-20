"""Track D "done when" check (PRD milestone M5 DoD): POST /forecast and the
CLI both produce identical Markdown/JSON output for the same ticker, when
wired to the same compiled graph."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from equity_ensemble.api.main import create_app
from equity_ensemble.cli.main import run_forecast_and_write
from equity_ensemble.data.fmp_client import RecordedFMPClient
from equity_ensemble.graph.build_graph import build_graph
from equity_ensemble.persistence.db import FakeDatabase
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
        generated_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
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


def _build_refusing_graph():
    """A graph whose nodes fail the test if ever invoked — proves a cache
    hit short-circuits before the pipeline runs at all, not just before
    the LLM call within it."""

    async def technicals(ticker, horizon_days):
        raise AssertionError("technicals ran on what should have been a cache hit")

    async def fundamentals(ticker, horizon_days):
        raise AssertionError("fundamentals ran on what should have been a cache hit")

    async def meta(ticker, horizon_days, technicals_claim, fundamentals_claim):
        raise AssertionError("meta_agent ran on what should have been a cache hit")

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


async def test_trace_url_is_null_when_langsmith_not_configured(monkeypatch):
    """No LANGSMITH_API_KEY (the default) must not change forecast behavior
    at all — tracing is purely opt-in."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.json()["trace_url"] is None


def test_build_langsmith_tracer_accepts_legacy_langchain_env_names(monkeypatch):
    """The underlying langsmith client accepts LANGSMITH_* or the legacy
    LANGCHAIN_* names (it tries LANGSMITH_ first, falls back to LANGCHAIN_)
    — our own gate has to agree, or it refuses to trace in a setup the
    client would have happily authenticated."""
    import equity_ensemble.api.main as api_main

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-legacy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "test-legacy-project")

    tracer = api_main._build_langsmith_tracer()

    assert tracer is not None
    assert tracer.project_name == "test-legacy-project"


def test_build_langsmith_tracer_prefers_langsmith_env_names(monkeypatch):
    import equity_ensemble.api.main as api_main

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "should-not-be-used")

    tracer = api_main._build_langsmith_tracer()

    assert tracer is not None
    assert tracer.project_name == "test-project"


def test_build_langsmith_tracer_is_none_with_no_key_configured(monkeypatch):
    import equity_ensemble.api.main as api_main

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    assert api_main._build_langsmith_tracer() is None


async def test_trace_url_is_populated_when_langsmith_tracer_available(monkeypatch):
    """When LangSmith is configured, the response carries this run's own
    trace URL — what the web UI's "View execution trace" link points at."""
    from langchain_core.callbacks.base import AsyncCallbackHandler

    import equity_ensemble.api.main as api_main

    class _FakeTracer(AsyncCallbackHandler):
        """A real callback handler (it gets attached to the actual graph
        invocation, same as the production tracer) that fakes just the
        trace-URL lookup."""

        def get_run_url(self) -> str:
            return "https://smith.langchain.com/o/fake-org/projects/p/fake-project/r/fake-run"

    monkeypatch.setattr(api_main, "_build_langsmith_tracer", lambda: _FakeTracer())

    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = api_main.create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.json()["trace_url"] == (
        "https://smith.langchain.com/o/fake-org/projects/p/fake-project/r/fake-run"
    )


async def test_trace_url_failure_does_not_break_an_otherwise_successful_forecast(monkeypatch):
    """A LangSmith hiccup (bad key, project not yet created) must degrade to
    trace_url=None, never fail the request that otherwise succeeded."""
    from langchain_core.callbacks.base import AsyncCallbackHandler

    import equity_ensemble.api.main as api_main

    class _FailingTracer(AsyncCallbackHandler):
        def get_run_url(self) -> str:
            raise RuntimeError("simulated LangSmith outage")

    monkeypatch.setattr(api_main, "_build_langsmith_tracer", lambda: _FailingTracer())

    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = api_main.create_app(graph=graph, fmp=fmp)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.json()["trace_url"] is None


async def test_fresh_forecast_via_api_is_persisted_and_marked_not_cached(monkeypatch):
    db = FakeDatabase()
    graph = _build_stub_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = create_app(graph=graph, fmp=fmp, db=db)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL", "horizon_days": 21})

    assert response.status_code == 200
    assert response.json()["cached"] is False
    # Persisted (PRD §4.2 audit trail) — checked directly rather than via
    # get_recent_forecast, since _canned_report()'s fixed timestamp is
    # deliberately outside the 24h window (age-window behavior itself is
    # covered by tests/unit/test_db.py and TestForecastCaching).
    assert await db.get_forecast(str(_CANNED_RUN_ID)) is not None


async def test_api_serves_a_cached_forecast_without_running_agents(monkeypatch):
    db = FakeDatabase()
    cached_report = _canned_report().model_copy(update={"generated_at": datetime.now(UTC)})
    await db.save_forecast(cached_report)

    graph = _build_refusing_graph()  # would fail the test if the pipeline ran
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)

    app = create_app(graph=graph, fmp=fmp, db=db)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/forecast", json={"ticker": "AAPL", "horizon_days": 21})

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert body["run_id"] == str(cached_report.run_id)
    # No graph invocation happened, so nothing was ever traced.
    assert body["trace_url"] is None


async def test_cli_serves_a_cached_forecast_without_running_agents(tmp_path, monkeypatch):
    db = FakeDatabase()
    cached_report = _canned_report().model_copy(update={"generated_at": datetime.now(UTC)})
    await db.save_forecast(cached_report)

    graph = _build_refusing_graph()
    fmp = RecordedFMPClient(FIXTURES_DIR / "fmp")

    async def fake_get_profile(ticker):
        return {"symbol": ticker}

    monkeypatch.setattr(fmp, "get_profile", fake_get_profile)
    monkeypatch.chdir(tmp_path)

    markdown = await run_forecast_and_write("AAPL", 21, graph=graph, fmp=fmp, db=db)

    assert cached_report.thesis in markdown
    assert (tmp_path / f"forecast_AAPL_{cached_report.run_id}.md").exists()


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
