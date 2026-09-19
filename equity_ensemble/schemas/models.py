"""Shared data contracts (PRD §7).

Every agent, the graph, the API/CLI, and persistence code depend on these
models. They are the interface boundary that makes the agent tracks in the
implementation plan buildable in parallel — changes here should go through
review since every downstream track is coded against these shapes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class Evidence(BaseModel):
    source: str
    date: date
    snippet: str


class AgentClaim(BaseModel):
    """Shared output shape for both specialist agents (PRD §7.1)."""

    agent: Literal["technicals", "fundamentals_sentiment"]
    ticker: str
    direction: Literal["bullish", "neutral", "bearish"]
    magnitude_bps: int
    confidence: float = Field(ge=0.0, le=1.0)

    # Technicals-only
    regime_label: str | None = None
    regime_changed: bool | None = None

    # Fundamentals/Sentiment-only
    sentiment_score: int | None = Field(default=None, ge=1, le=100)
    key_risk_flags: list[str] = Field(default_factory=list)

    evidence: list[Evidence] = Field(min_length=1)
    falsifiers: list[str] = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def ticker_is_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError("ticker must be uppercase")
        return v


class RetrievalChunk(BaseModel):
    """A single chunk of a filing or news article prepared for pgvector search
    (PRD §7.2)."""

    chunk_id: str
    ticker: str
    source_type: Literal["10-K", "10-Q", "8-K", "news"]
    source_date: date
    section: str | None = None
    text: str
    embedding: list[float] = Field(min_length=1)


class Distribution(BaseModel):
    bullish_pct: float
    neutral_pct: float
    bearish_pct: float

    @model_validator(mode="after")
    def sums_to_100(self) -> Distribution:
        total = self.bullish_pct + self.neutral_pct + self.bearish_pct
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"distribution must sum to 100, got {total}")
        return self


class ForecastReport(BaseModel):
    """Meta-Agent output — the deliverable (PRD §7.3)."""

    run_id: UUID
    ticker: str
    horizon_days: int = 21
    generated_at: datetime

    distribution: Distribution
    recommendation: Literal["bullish_lean", "neutral", "bearish_lean", "requires_review"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    escalate_to_analyst: bool
    escalation_reason: str | None = None

    thesis: str
    agent_claims: list[AgentClaim] = Field(min_length=1)
    critique_log: list[str] = Field(default_factory=list)
    citations: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def escalation_reason_required_when_escalating(self) -> ForecastReport:
        if self.escalate_to_analyst and not self.escalation_reason:
            raise ValueError("escalation_reason is required when escalate_to_analyst is true")
        return self
