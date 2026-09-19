"""Contract stability tests for equity_ensemble.schemas.models.

Every Phase 1 track codes against these schemas — run this whenever
models.py changes, since a shape change here can silently break every
track's fixtures.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from equity_ensemble.schemas.models import (
    AgentClaim,
    Distribution,
    Evidence,
    ForecastReport,
)


def _evidence() -> Evidence:
    return Evidence(source="10-K Risk Factors", date=date(2026, 6, 1), snippet="...")


class TestAgentClaim:
    def test_valid_claim_round_trips(self, technicals_claim_factory):
        claim = technicals_claim_factory()
        assert claim.direction == "bullish"

    def test_ticker_must_be_uppercase(self, technicals_claim_factory):
        with pytest.raises(ValidationError):
            technicals_claim_factory(ticker="aapl")

    def test_evidence_must_be_non_empty(self, technicals_claim_factory):
        with pytest.raises(ValidationError):
            technicals_claim_factory(evidence=[])

    def test_falsifiers_must_be_non_empty(self, technicals_claim_factory):
        with pytest.raises(ValidationError):
            technicals_claim_factory(falsifiers=[])

    def test_confidence_out_of_range_rejected(self, technicals_claim_factory):
        with pytest.raises(ValidationError):
            technicals_claim_factory(confidence=1.5)


class TestDistribution:
    def test_valid_distribution_sums_to_100(self):
        Distribution(bullish_pct=40, neutral_pct=30, bearish_pct=30)

    def test_distribution_not_summing_to_100_rejected(self):
        with pytest.raises(ValidationError):
            Distribution(bullish_pct=50, neutral_pct=30, bearish_pct=30)


class TestForecastReport:
    def _report_kwargs(self, **overrides):
        defaults = dict(
            run_id=uuid4(),
            ticker="AAPL",
            horizon_days=21,
            generated_at=datetime(2026, 9, 19, 12, 0, 0),
            distribution=Distribution(bullish_pct=40, neutral_pct=30, bearish_pct=30),
            recommendation="bullish_lean",
            overall_confidence=0.6,
            escalate_to_analyst=False,
            escalation_reason=None,
            thesis="Sample thesis.",
            agent_claims=[
                AgentClaim(
                    agent="technicals",
                    ticker="AAPL",
                    direction="bullish",
                    magnitude_bps=150,
                    confidence=0.7,
                    regime_label="Trending Bull",
                    regime_changed=False,
                    evidence=[_evidence()],
                    falsifiers=["ADX reverses"],
                )
            ],
            critique_log=[],
            citations=[_evidence()],
        )
        defaults.update(overrides)
        return defaults

    def test_valid_report_round_trips(self):
        ForecastReport(**self._report_kwargs())

    def test_recommendation_excludes_trade_language(self):
        with pytest.raises(ValidationError):
            ForecastReport(**self._report_kwargs(recommendation="buy"))

    def test_escalation_reason_required_when_escalating(self):
        with pytest.raises(ValidationError):
            ForecastReport(
                **self._report_kwargs(escalate_to_analyst=True, escalation_reason=None)
            )

    def test_escalation_reason_present_is_valid(self):
        ForecastReport(
            **self._report_kwargs(
                escalate_to_analyst=True,
                escalation_reason="Material disagreement unresolved after critique.",
                recommendation="requires_review",
            )
        )
