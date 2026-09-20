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
from equity_ensemble.data.retrieval import (
    Embedder,
    PendingChunk,
    embed_chunks,
    plan_chunks,
    rank_chunks,
)
from equity_ensemble.llm.client import LLMClient
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim, RetrievalChunk

logger = logging.getLogger(__name__)

NEWS_LOOKBACK_DAYS = 14
RETRIEVAL_CANDIDATE_LIMIT = 10

# Bounds embedding-API volume per run (PRD §5.1's cost-visibility concern in
# practice: Voyage's no-payment-method free tier is 10K tokens/minute, and a
# single 10-Q can chunk into 30+ pieces on its own). Measured against a real
# AAPL 10-Q with Voyage's own tokenizer (voyageai.Client().count_tokens):
# financial filing text runs ~1,100 tokens per 500-word chunk (roughly 2.2x
# a plain-English word count, not the ~1.3x a naive estimate would assume —
# dense numbers/terminology tokenize less efficiently). 6 chunks measured at
# 7,331 tokens, ~27% under the cap; a denser filer could still occasionally
# exceed it, in which case the run degrades to the standard 503, not a crash.
# This bounds tokens only. The same tier caps *requests* at 3/minute, which
# `_ingest_filings` handles by embedding its whole batch in one call.
MAX_CHUNKS_PER_RUN = 6

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


def _select_relevant_filings(filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PRD §6.2 Inputs: 'Most recent 10-K and any 10-Q/8-K filed since it' —
    not every filing FMP's lookback window returns. Keeps ingestion (and
    embedding volume) bounded to what the claim actually needs rather than
    growing with however wide the vendor's search window happens to be.

    Returned most-recent-first: `_ingest_filings` spends its chunk budget in
    this order, and a recent 8-K (a surprise executive departure, a
    guidance cut) is exactly the kind of disclosure PRD §6.2 wants this
    agent to catch — it shouldn't lose out on the embedding budget to the
    much larger, less time-sensitive 10-K just because the 10-K is older.
    """
    if not filings:
        return []
    by_date_desc = sorted(filings, key=lambda f: _parse_date(f["date"]), reverse=True)
    ten_ks = [f for f in by_date_desc if f.get("type") == "10-K"]
    if not ten_ks:
        return by_date_desc
    cutoff = _parse_date(ten_ks[0]["date"])  # most recent 10-K comes first, descending
    return [f for f in by_date_desc if _parse_date(f["date"]) >= cutoff]


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
        """Plans chunks across every filing first, then embeds what's missing
        in one call.

        Two properties matter for staying inside Voyage's free tier, which
        caps requests per minute as well as tokens. Planning across all
        filings before embedding makes the whole ingest one request instead
        of one per filing. Filtering against chunks already stored makes a
        repeat forecast for the same ticker cost zero requests, since the
        filing text and its deterministic `chunk_id`s have not changed.
        """
        pending: list[PendingChunk] = []
        remaining = MAX_CHUNKS_PER_RUN
        for filing in filings:
            if remaining <= 0:
                break
            text = (filing.get("text") or "").strip()
            if not text:
                continue
            planned = plan_chunks(
                ticker=ticker,
                source_type=filing["type"],
                source_date=_parse_date(filing["date"]),
                section=filing.get("section"),
                text=text,
                max_chunks=remaining,
            )
            pending.extend(planned)
            remaining -= len(planned)

        # The budget deliberately isn't re-spent on further filings when
        # chunks are skipped: reusing it would mean a warm ticker embeds six
        # *new* chunks on every run and never gets cheap.
        already_stored = await self.db.existing_chunk_ids([p.chunk_id for p in pending])
        new_chunks = await embed_chunks(
            [p for p in pending if p.chunk_id not in already_stored], embedder=self.embedder
        )
        await self.db.save_chunks(new_chunks)
        log_event(
            logger,
            "fundamentals_sentiment",
            "thinking",
            ticker=ticker,
            step="filing_ingest",
            planned=len(pending),
            embedded=len(new_chunks),
            reused=len(already_stored),
        )

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
        await self._ingest_filings(ticker, _select_relevant_filings(filings))

        query_text = _build_retrieval_query(ticker, news_scores)
        [query_embedding] = await self.embedder.embed([query_text], input_type="query")
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
