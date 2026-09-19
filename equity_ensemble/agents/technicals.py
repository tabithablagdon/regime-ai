"""Technicals Agent — owned by Track A (PRD §6.1).

Work items (see implementation plan, Track A):
1. Rule-based regime classifier (pure function, not the LLM): realized
   volatility + ADX trend strength + price-vs-moving-average ->
   {Trending Bull, Trending Bear, Mean-Reverting, Volatile/Choppy}.
2. FMPClient methods for OHLCV + technical indicators (data/fmp_client.py).
3. ReAct loop: read last regime from Database.get_last_regime -> fetch
   today's price/indicators -> detect transition -> LLM narrates the
   computed regime into an AgentClaim (the LLM never computes the regime).
4. Bounded retry via agents.base.run_with_validation_retry.
"""

from __future__ import annotations

from equity_ensemble.agents.base import BaseSpecialistAgent
from equity_ensemble.data.fmp_client import FMPClient
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim

REGIME_LABELS = ("Trending Bull", "Trending Bear", "Mean-Reverting", "Volatile/Choppy")


def classify_regime(
    *,
    realized_volatility: float,
    adx: float,
    price_vs_moving_average_pct: float,
) -> str:
    """Pure, deterministic regime classifier (PRD §6.1). No LLM call.

    Track A: implement the actual thresholds and unit-test against
    synthetic OHLCV series covering each of REGIME_LABELS plus edge cases
    (transition detected, no transition, insufficient history).
    """
    raise NotImplementedError("Track A: implement regime classification thresholds")


class TechnicalsAgent(BaseSpecialistAgent[AgentClaim]):
    def __init__(self, llm, fmp: FMPClient, db: Database) -> None:
        super().__init__(llm)
        self.fmp = fmp
        self.db = db

    async def run(self, ticker: str, horizon_days: int) -> AgentClaim:
        raise NotImplementedError("Track A: implement the ReAct loop described in PRD §6.1")
