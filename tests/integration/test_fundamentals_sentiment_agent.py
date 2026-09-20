"""Track B "done when" check: given a ticker, against recorded filing/news
fixtures and FakeEmbedder, FundamentalsSentimentAgent returns a schema-valid
AgentClaim with sentiment_score, key_risk_flags, and non-empty cited
evidence — without any live FMP, Voyage, or LLM access."""

from __future__ import annotations

import logging
from datetime import date
from typing import NamedTuple

from equity_ensemble.agents.fundamentals_sentiment import (
    MAX_CHUNKS_PER_RUN,
    FundamentalsSentimentAgent,
    _NewsSentimentItem,
    _NewsSentimentResponse,
)
from equity_ensemble.data.retrieval import FakeEmbedder
from equity_ensemble.llm.client import FakeLLMClient
from equity_ensemble.schemas.models import AgentClaim, Evidence


class _EmbedCall(NamedTuple):
    texts: list[str]
    input_type: str


class _CountingEmbedder:
    """`FakeEmbedder` that records every call it receives.

    Lets tests assert on embedding *request count*, which `FakeEmbedder`
    alone cannot see. That is the dimension Voyage's free tier meters at 3
    per minute, and getting it wrong cost a 60s backoff stall in a live run
    rather than a visible failure — so it needs a test that would catch the
    regression instead of just slowing down.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._inner = FakeEmbedder(dimension=dimension)
        self.calls: list[_EmbedCall] = []

    async def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        self.calls.append(_EmbedCall(list(texts), input_type))
        return await self._inner.embed(texts, input_type=input_type)

    def calls_of(self, input_type: str) -> list[_EmbedCall]:
        return [call for call in self.calls if call.input_type == input_type]


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


async def test_returns_schema_valid_claim(recorded_fmp, fake_db, caplog):
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)
    caplog.set_level(logging.INFO)

    claim = await agent.run("AAPL", horizon_days=21)

    assert isinstance(claim, AgentClaim)
    assert claim.agent == "fundamentals_sentiment"
    assert claim.ticker == "AAPL"
    assert claim.sentiment_score == 35
    assert claim.key_risk_flags == ["guidance_cut"]
    assert len(claim.evidence) >= 1
    assert "fundamentals_sentiment called ticker=AAPL horizon_days=21" in caplog.text
    assert "step=news_sentiment" in caplog.text
    assert "step=filing_retrieval" in caplog.text
    assert "fundamentals_sentiment decision ticker=AAPL" in caplog.text


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


async def test_embedding_volume_is_capped_across_all_filings(recorded_fmp, fake_db, monkeypatch):
    """Regression test for the Voyage free-tier rate limit hit in practice:
    a run with several large filings must never embed more than
    MAX_CHUNKS_PER_RUN chunks total, no matter how much text FMP returns."""
    huge_text = " ".join(f"w{i}" for i in range(5000))  # far more than one filing's worth

    async def many_large_filings(ticker):
        return [
            {"type": "10-K", "date": "2025-10-01", "section": None, "text": huge_text},
            {"type": "8-K", "date": "2025-11-01", "section": None, "text": huge_text},
            {"type": "10-Q", "date": "2026-01-15", "section": None, "text": huge_text},
        ]

    monkeypatch.setattr(recorded_fmp, "get_filings", many_large_filings)
    embedder = FakeEmbedder(dimension=64)
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    await agent.run("AAPL", horizon_days=21)

    stored = await fake_db.search_chunks("AAPL", [0.0] * 64, limit=100)
    assert len(stored) == MAX_CHUNKS_PER_RUN
    # budget spent on the most recent filing first, not the oldest
    assert stored[0].source_type == "10-Q"


async def test_all_filings_are_embedded_in_one_request(recorded_fmp, fake_db, monkeypatch):
    """Voyage's free tier caps requests at 3/minute, so embedding one request
    per filing exhausted the quota on filing count alone — three filings plus
    the query made four requests and bought a 60s backoff stall. The whole
    ingest must be a single request however many filings FMP returns."""
    text = " ".join(f"w{i}" for i in range(1200))

    async def three_filings(ticker):
        return [
            {"type": "10-K", "date": "2025-10-01", "section": None, "text": text},
            {"type": "8-K", "date": "2025-11-01", "section": None, "text": text},
            {"type": "10-Q", "date": "2026-01-15", "section": None, "text": text},
        ]

    monkeypatch.setattr(recorded_fmp, "get_filings", three_filings)
    embedder = _CountingEmbedder()
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    await agent.run("AAPL", horizon_days=21)

    document_batches = embedder.calls_of("document")
    assert len(document_batches) == 1
    assert len(document_batches[0].texts) == MAX_CHUNKS_PER_RUN
    assert len(embedder.calls) == 2  # the one chunk batch, plus the query


async def test_repeat_run_for_same_ticker_embeds_no_filing_text(recorded_fmp, fake_db, caplog):
    """`chunk_id` is deterministic and `save_chunks` is ON CONFLICT DO
    NOTHING, so re-embedding a filing already stored pays the vendor for
    vectors Postgres discards. A warm ticker must spend nothing on chunks."""
    embedder = _CountingEmbedder()
    llm = FakeLLMClient(
        responses=[
            _scripted_news_scores(),
            _scripted_claim(),
            _scripted_news_scores(),
            _scripted_claim(),
        ]
    )
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    await agent.run("AAPL", horizon_days=21)
    assert embedder.calls_of("document"), "first run should embed the filing text"
    calls_after_cold_run = len(embedder.calls)

    caplog.set_level(logging.INFO)
    await agent.run("AAPL", horizon_days=21)

    second_run = embedder.calls[calls_after_cold_run:]
    assert [call.input_type for call in second_run] == ["query"]
    assert "step=filing_ingest" in caplog.text
    assert "embedded=0" in caplog.text


async def test_retrieval_query_is_embedded_as_a_query(recorded_fmp, fake_db):
    """voyage-finance-2 is asymmetric, so the query has to be embedded as a
    query; sending it as a document compares it against the stored chunks on
    the wrong footing and quietly degrades relevance."""
    embedder = _CountingEmbedder()
    llm = FakeLLMClient(responses=[_scripted_news_scores(), _scripted_claim()])
    agent = FundamentalsSentimentAgent(llm, recorded_fmp, fake_db, embedder)

    await agent.run("AAPL", horizon_days=21)

    query_calls = embedder.calls_of("query")
    assert len(query_calls) == 1
    assert query_calls[0].texts == ["AAPL: AAPL guidance cut spooks investors ahead of earnings"]


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
