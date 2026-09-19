"""Fundamentals/Sentiment Agent — Track B (PRD §6.2).

Sequential tool-invocation loop: fetch news -> score sentiment per article
-> ingest the current filing (chunk/embed/store) -> retrieve the chunks
relevant to the claim being formed -> only then compute the aggregate
claim, so a strong qualitative disclosure can override a purely
price-momentum-driven initial read.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from equity_ensemble.agents.base import BaseSpecialistAgent, run_with_validation_retry
from equity_ensemble.agents.trace import log_event, summarize_claim, summarize_evidence
from equity_ensemble.data.fmp_client import FMPClient
from equity_ensemble.data.retrieval import Embedder, build_chunks, rank_chunks
from equity_ensemble.llm.client import LLMClient
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim, RetrievalChunk

logger = logging.getLogger(__name__)

NEWS_LOOKBACK_DAYS = 14
RETRIEVAL_CANDIDATE_LIMIT = 10

NEWS_SENTIMENT_SYSTEM_PROMPT = (
    "You are a financial news sentiment scorer. Score each article's sentiment "
    "toward the company on a 1-100 scale (1=very negative, 50=neutral, "
    "100=very positive), based only on the title and snippet given."
)

FUNDAMENTALS_SYSTEM_PROMPT = (
    "You are the Fundamentals/Sentiment specialist in an equity forecast "
    "ensemble. Ground every claim in the news sentiment scores and retrieved "
    "filing evidence provided below — never in parametric knowledge about the "
    "company. Every evidence[] entry must cite a specific source (filing "
    "section + date, or article title + date) taken from what's shown to you. "
    "If a disclosure materially contradicts the news-driven read, let it "
    "override your initial impression."
)


class _NewsSentimentItem(BaseModel):
    title: str
    published_date: str
    sentiment: int = Field(ge=1, le=100)


class _NewsSentimentResponse(BaseModel):
    items: list[_NewsSentimentItem] = Field(default_factory=list)


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _build_retrieval_query(ticker: str, news_scores: list[_NewsSentimentItem]) -> str:
    """Focus retrieval on the most negative headline, if any — that's the
    disclosure most likely to need filing-level grounding or rebuttal."""
    if not news_scores:
        return f"{ticker} recent risk factors and material developments"
    most_negative = min(news_scores, key=lambda item: item.sentiment)
    return f"{ticker}: {most_negative.title}"


def _build_claim_prompt(
    ticker: str,
    news_scores: list[_NewsSentimentItem],
    top_chunks: list[RetrievalChunk],
    estimate_revisions: list[dict[str, Any]],
) -> str:
    lines = [f"Ticker: {ticker}"]

    if news_scores:
        lines.append("News sentiment scores (1-100, 100 = very positive):")
        lines += [
            f"- [{item.published_date}] {item.title}: {item.sentiment}" for item in news_scores
        ]
    else:
        lines.append("No recent news found in the last window.")

    if top_chunks:
        lines.append("Retrieved filing evidence:")
        lines += [
            f"- [{c.source_type} {c.source_date}, {c.section or 'n/a'}] {c.text[:400]}"
            for c in top_chunks
        ]
    else:
        lines.append("No filing chunks retrieved.")

    if estimate_revisions:
        lines.append(f"Consensus estimate revisions available: {len(estimate_revisions)} entries.")

    lines.append(
        "Using only the evidence above, produce your claim. Every evidence[] entry must cite "
        "a specific source (filing section + date, or article title + date) from what's shown here."
    )
    return "\n".join(lines)


class FundamentalsSentimentAgent(BaseSpecialistAgent[AgentClaim]):
    def __init__(self, llm: LLMClient, fmp: FMPClient, db: Database, embedder: Embedder) -> None:
        super().__init__(llm)
        self.fmp = fmp
        self.db = db
        self.embedder = embedder

    async def _score_news(
        self, ticker: str, articles: list[dict[str, Any]]
    ) -> list[_NewsSentimentItem]:
        if not articles:
            log_event(
                logger,
                "fundamentals_sentiment",
                "thinking",
                ticker=ticker,
                step="news_sentiment",
                article_count=0,
                scores=[],
            )
            return []
        article_lines = "\n".join(
            f"- [{a.get('publishedDate', 'unknown date')}] {a.get('title', '')}: "
            f"{(a.get('text') or '')[:300]}"
            for a in articles
        )
        response = await self.llm.complete_structured(
            system_prompt=NEWS_SENTIMENT_SYSTEM_PROMPT,
            user_prompt=f"Ticker: {ticker}\n\n{article_lines}",
            response_model=_NewsSentimentResponse,
        )
        log_event(
            logger,
            "fundamentals_sentiment",
            "thinking",
            ticker=ticker,
            step="news_sentiment",
            article_count=len(articles),
            scores=[f"{item.title}:{item.sentiment}" for item in response.items],
        )
        return response.items

    async def _ingest_filings(self, ticker: str, filings: list[dict[str, Any]]) -> None:
        for filing in filings:
            text = (filing.get("text") or "").strip()
            if not text:
                continue
            chunks = await build_chunks(
                ticker=ticker,
                source_type=filing["type"],
                source_date=_parse_date(filing["date"]),
                section=filing.get("section"),
                text=text,
                embedder=self.embedder,
            )
            if chunks:
                await self.db.save_chunks(chunks)

    async def run(self, ticker: str, horizon_days: int) -> AgentClaim:
        ticker = ticker.upper()
        log_event(
            logger,
            "fundamentals_sentiment",
            "called",
            ticker=ticker,
            horizon_days=horizon_days,
        )

        news_articles = await self.fmp.get_news(ticker, days=NEWS_LOOKBACK_DAYS)
        news_scores = await self._score_news(ticker, news_articles)

        filings = await self.fmp.get_filings(ticker)
        await self._ingest_filings(ticker, filings)

        query_text = _build_retrieval_query(ticker, news_scores)
        [query_embedding] = await self.embedder.embed([query_text])
        candidates = await self.db.search_chunks(
            ticker, query_embedding, limit=RETRIEVAL_CANDIDATE_LIMIT
        )
        top_chunks = rank_chunks(query_embedding, candidates)
        log_event(
            logger,
            "fundamentals_sentiment",
            "thinking",
            ticker=ticker,
            step="filing_retrieval",
            query=query_text,
            candidate_count=len(candidates),
            top_chunks=[
                f"{chunk.source_type}:{chunk.section or 'n/a'}({chunk.source_date.isoformat()})"
                for chunk in top_chunks
            ],
        )

        try:
            estimate_revisions = await self.fmp.get_estimate_revisions(ticker)
        except NotImplementedError:
            estimate_revisions = []

        def build_user_prompt(previous_error: str | None) -> str:
            prompt = _build_claim_prompt(ticker, news_scores, top_chunks, estimate_revisions)
            if previous_error:
                prompt += (
                    f"\n\nYour previous response failed schema validation: {previous_error}\n"
                    "Fix it and return a valid claim."
                )
            return prompt

        claim = await run_with_validation_retry(
            llm=self.llm,
            system_prompt=FUNDAMENTALS_SYSTEM_PROMPT,
            build_user_prompt=build_user_prompt,
            response_model=AgentClaim,
            agent="fundamentals_sentiment",
            ticker=ticker,
        )
        claim = claim.model_copy(update={"agent": "fundamentals_sentiment", "ticker": ticker})
        log_event(
            logger,
            "fundamentals_sentiment",
            "decision",
            ticker=ticker,
            claim=summarize_claim(claim),
            falsifiers=claim.falsifiers,
            evidence=summarize_evidence(claim),
        )
        return claim
