"""Fundamentals/Sentiment Agent — owned by Track B (PRD §6.2).

Work items (see implementation plan, Track B):
1. Sequential tool-invocation loop: fetch news -> score sentiment per
   article -> retrieve relevant filing chunks (data/retrieval.py) for the
   specific claim being formed -> aggregate.
2. FMPClient methods for filings/news (data/fmp_client.py), with the SEC
   EDGAR full-text search fallback when a filing isn't mirrored by FMP.
3. Every AgentClaim.evidence entry must carry a real citation — enforced by
   the Pydantic schema (schemas/models.py), not just requested in the prompt.
"""

from __future__ import annotations

from equity_ensemble.agents.base import BaseSpecialistAgent
from equity_ensemble.data.fmp_client import FMPClient
from equity_ensemble.data.retrieval import Embedder
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim


class FundamentalsSentimentAgent(BaseSpecialistAgent[AgentClaim]):
    def __init__(self, llm, fmp: FMPClient, db: Database, embedder: Embedder) -> None:
        super().__init__(llm)
        self.fmp = fmp
        self.db = db
        self.embedder = embedder

    async def run(self, ticker: str, horizon_days: int) -> AgentClaim:
        raise NotImplementedError(
            "Track B: implement the sequential tool-invocation loop, PRD §6.2"
        )
