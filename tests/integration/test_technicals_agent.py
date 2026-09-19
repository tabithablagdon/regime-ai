"""Track A "done when" check: given a ticker, against RecordedFMPClient
fixtures and FakeLLMClient, TechnicalsAgent returns a schema-valid
AgentClaim with regime_label set — without any live FMP or LLM access."""

from __future__ import annotations

from datetime import date

from equity_ensemble.agents.technicals import TechnicalsAgent
from equity_ensemble.llm.client import FakeLLMClient
from equity_ensemble.schemas.models import AgentClaim, Evidence


def _scripted_claim() -> AgentClaim:
    return AgentClaim(
        agent="technicals",
        ticker="PLACEHOLDER",
        direction="neutral",
        magnitude_bps=10,
        confidence=0.5,
        regime_label="placeholder",
        regime_changed=False,
        evidence=[
            Evidence(source="today's OHLCV + indicators", date=date(2026, 8, 1), snippet="n/a")
        ],
        falsifiers=["placeholder"],
    )


async def test_returns_schema_valid_claim_with_computed_regime(recorded_fmp, fake_db):
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = TechnicalsAgent(llm, recorded_fmp, fake_db)

    claim = await agent.run("AAPL", horizon_days=21)

    assert isinstance(claim, AgentClaim)
    assert claim.agent == "technicals"
    assert claim.ticker == "AAPL"
    # AAPL fixture is a steady uptrend with ADX=32.5 (trending) and price well
    # above its 50-day SMA, so the deterministic classifier must say Bull —
    # regardless of what the (irrelevant, scripted) LLM response claimed.
    assert claim.regime_label == "Trending Bull"


async def test_no_prior_regime_means_no_transition(recorded_fmp, fake_db):
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = TechnicalsAgent(llm, recorded_fmp, fake_db)

    claim = await agent.run("AAPL", horizon_days=21)

    assert claim.regime_changed is False


async def test_detects_regime_transition_from_memory(recorded_fmp, fake_db):
    await fake_db.save_regime("AAPL", date(2026, 1, 1), "Mean-Reverting")
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = TechnicalsAgent(llm, recorded_fmp, fake_db)

    claim = await agent.run("AAPL", horizon_days=21)

    assert claim.regime_label == "Trending Bull"
    assert claim.regime_changed is True


async def test_persists_regime_for_next_run(recorded_fmp, fake_db):
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = TechnicalsAgent(llm, recorded_fmp, fake_db)

    await agent.run("AAPL", horizon_days=21)

    assert await fake_db.get_last_regime("AAPL") == "Trending Bull"


async def test_lowercase_ticker_is_normalized(recorded_fmp, fake_db):
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = TechnicalsAgent(llm, recorded_fmp, fake_db)

    claim = await agent.run("aapl", horizon_days=21)

    assert claim.ticker == "AAPL"
