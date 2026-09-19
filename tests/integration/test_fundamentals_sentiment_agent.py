"""Track B "done when" check: given a ticker, against recorded filing/news
fixtures and FakeEmbedder, FundamentalsSentimentAgent returns a schema-valid
AgentClaim with sentiment_score, key_risk_flags, and non-empty cited
evidence — without any live FMP, Voyage, or LLM access."""

from __future__ import annotations

from datetime import date

from equity_ensemble.agents.fundamentals_sentiment import (
    FundamentalsSentimentAgent,
    _NewsSentimentItem,
    _NewsSentimentResponse,
)
from equity_ensemble.data.retrieval import FakeEmbedder
from equity_ensemble.llm.client import FakeLLMClient
from equity_ensemble.schemas.models import AgentClaim, Evidence


def _scripted_news_scores() -> _NewsSentimentResponse:
    return _NewsSentimentResponse(
        items=[
            _NewsSentimentItem(
                title="AAPL guidance cut spooks investors ahead of earnings",
                published_date="2026-08-20",
                sentiment=20,
            ),
            _NewsSentimentItem(
                title="AAPL announces expanded services partnership",
                published_date="2026-08-15",
                sentiment=65,
            ),
        ]
    )


def _scripted_claim() -> AgentClaim:
    return AgentClaim(
        agent="fundamentals_sentiment",
        ticker="PLACEHOLDER",
        direction="bearish",
        magnitude_bps=180,
        confidence=0.7,
        sentiment_score=35,
        key_risk_flags=["guidance_cut"],
        evidence=[
            Evidence(
                source="10-Q Risk Factors",
                date=date(2026, 8, 1),
                snippet="lowered its guidance for the current quarter",
            )
        ],
        falsifiers=["Next 8-K reaffirms prior guidance"],
    )


async def test_returns_schema_valid_claim(recorded_fmp, fake_db):
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    claim = await agent.run("AAPL", horizon_days=21)

    assert isinstance(claim, AgentClaim)
    assert claim.agent == "fundamentals_sentiment"
    assert claim.ticker == "AAPL"
    assert claim.sentiment_score == 35
    assert claim.key_risk_flags == ["guidance_cut"]
    assert len(claim.evidence) >= 1


async def test_ingests_filing_chunks_into_the_database(recorded_fmp, fake_db):
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    await agent.run("AAPL", horizon_days=21)

    stored = await fake_db.search_chunks("AAPL", [0.0] * 64, limit=10)
    assert len(stored) >= 1
    assert stored[0].source_type == "10-Q"


async def test_lowercase_ticker_is_normalized(recorded_fmp, fake_db):
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    claim = await agent.run("aapl", horizon_days=21)

    assert claim.ticker == "AAPL"


async def test_no_news_falls_back_to_generic_retrieval_query(recorded_fmp, fake_db, monkeypatch):
    async def empty_news(ticker, days=14):
        return []

    monkeypatch.setattr(recorded_fmp, "get_news", empty_news)
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    claim = await agent.run("AAPL", horizon_days=21)

    assert isinstance(claim, AgentClaim)
    assert len(llm.calls) == 1  # no news -> no sentiment-scoring call, only the claim call
