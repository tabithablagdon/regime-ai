from __future__ import annotations

import logging

import pytest

from equity_ensemble.agents.meta_agent import (
    MetaAgent,
    _ThesisResponse,
    check_conflict,
    fit_distribution,
    resolve_weights,
)
from equity_ensemble.llm.client import FakeLLMClient


class TestCheckConflict:
    def test_opposed_and_both_confident_is_conflict(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.7)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.6)
        assert check_conflict(t, f) is True

    def test_opposed_but_low_confidence_is_not_conflict(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.4)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.6)
        assert check_conflict(t, f) is False

    def test_neutral_vs_bearish_is_not_conflict(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="neutral", confidence=0.9)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.9)
        assert check_conflict(t, f) is False

    def test_agreement_is_not_conflict(self, technicals_claim_factory, fundamentals_claim_factory):
        t = technicals_claim_factory(direction="bullish", confidence=0.8)
        f = fundamentals_claim_factory(direction="bullish", confidence=0.8)
        assert check_conflict(t, f) is False


class TestResolveWeights:
    def test_known_regime(self):
        weights = resolve_weights("Volatile/Choppy")
        assert weights["technicals"] == 0.6
        assert weights["fundamentals_sentiment"] == 0.4

    def test_unknown_regime_falls_back_to_default(self):
        weights = resolve_weights("Some Regime Not In Table")
        assert weights == resolve_weights(None)

    def test_none_regime_uses_default(self):
        weights = resolve_weights(None)
        assert weights["technicals"] == 0.5


class TestFitDistribution:
    def test_distribution_sums_to_100(self, technicals_claim_factory, fundamentals_claim_factory):
        t = technicals_claim_factory(direction="bullish", magnitude_bps=200, confidence=0.9)
        f = fundamentals_claim_factory(direction="bullish", magnitude_bps=200, confidence=0.9)
        weights = resolve_weights("Trending Bull")
        dist = fit_distribution(t, f, weights)
        total = dist.bullish_pct + dist.neutral_pct + dist.bearish_pct
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_strong_bullish_agreement_is_mostly_bullish(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", magnitude_bps=400, confidence=0.95)
        f = fundamentals_claim_factory(direction="bullish", magnitude_bps=400, confidence=0.95)
        weights = resolve_weights("Trending Bull")
        dist = fit_distribution(t, f, weights)
        assert dist.bullish_pct > dist.bearish_pct
        assert dist.bullish_pct > 50

    def test_single_available_claim_still_produces_valid_distribution(
        self, technicals_claim_factory
    ):
        t = technicals_claim_factory(direction="bearish", magnitude_bps=300, confidence=0.9)
        weights = resolve_weights(t.regime_label)
        dist = fit_distribution(t, None, weights)
        total = dist.bullish_pct + dist.neutral_pct + dist.bearish_pct
        assert total == pytest.approx(100.0, abs=1e-6)
        assert dist.bearish_pct > dist.bullish_pct

    def test_requires_at_least_one_claim(self):
        with pytest.raises(ValueError):
            fit_distribution(None, None, resolve_weights(None))


class TestMetaAgentRun:
    async def test_agreement_case_no_critique(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.8)
        f = fundamentals_claim_factory(direction="bullish", confidence=0.8)
        llm = FakeLLMClient(responses=[_ThesisResponse(thesis="Both agents agree on upside.")])

        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, t, f)

        assert report.critique_log == []
        assert report.escalate_to_analyst is False
        assert report.agent_claims == [t, f]
        assert len(llm.calls) == 1  # rationale only, no critique round

    async def test_conflict_triggers_single_critique_round(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.8)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.8)

        # Post-critique: both converge to neutral/low-confidence agreement.
        revised_t = technicals_claim_factory(direction="neutral", confidence=0.5)
        revised_f = fundamentals_claim_factory(direction="neutral", confidence=0.5)
        thesis = _ThesisResponse(thesis="Agents converged after reviewing counter-evidence.")

        llm = FakeLLMClient(responses=[revised_t, revised_f, thesis])
        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, t, f)

        assert len(report.critique_log) == 4
        # dissent log always shows the original, unmodified claims
        assert report.agent_claims == [t, f]
        assert len(llm.calls) == 3

    async def test_unresolved_conflict_after_critique_escalates(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.9)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.9)

        # Post-critique: both specialists stick to their guns.
        revised_t = technicals_claim_factory(direction="bullish", confidence=0.9)
        revised_f = fundamentals_claim_factory(direction="bearish", confidence=0.9)
        thesis = _ThesisResponse(thesis="Agents remain in disagreement; recommend review.")

        llm = FakeLLMClient(responses=[revised_t, revised_f, thesis])
        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, t, f)

        assert report.escalate_to_analyst is True
        assert report.recommendation == "requires_review"
        assert report.escalation_reason is not None

    async def test_single_agent_unavailable_forces_escalation(self, fundamentals_claim_factory):
        f = fundamentals_claim_factory(direction="bearish", confidence=0.8)
        thesis = _ThesisResponse(thesis="Only the fundamentals/sentiment agent reported.")
        llm = FakeLLMClient(responses=[thesis])

        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, None, f)

        assert report.escalate_to_analyst is True
        assert "unavailable" in report.escalation_reason
        assert report.agent_claims == [f]

    async def test_citations_are_deduplicated(
        self, technicals_claim_factory, fundamentals_claim_factory
    ):
        from datetime import date

        from equity_ensemble.schemas.models import Evidence

        shared_evidence = Evidence(source="Shared 10-Q", date=date(2026, 8, 1), snippet="shared")
        t = technicals_claim_factory(
            direction="bullish", confidence=0.6, evidence=[shared_evidence]
        )
        f = fundamentals_claim_factory(
            direction="bullish", confidence=0.6, evidence=[shared_evidence]
        )
        llm = FakeLLMClient(responses=[_ThesisResponse(thesis="Agreement.")])

        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, t, f)

        assert len(report.citations) == 1

    async def test_logs_call_thinking_and_decision(
        self, technicals_claim_factory, fundamentals_claim_factory, caplog
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.8)
        f = fundamentals_claim_factory(direction="bullish", confidence=0.8)
        thesis = "Both agents agree on upside."
        llm = FakeLLMClient(responses=[_ThesisResponse(thesis=thesis)])

        caplog.set_level(logging.INFO)
        meta = MetaAgent(llm)
        report = await meta.run("AAPL", 21, t, f)

        assert "meta_agent called ticker=AAPL" in caplog.text
        assert "step=conflict_check" in caplog.text
        assert "conflict=false" in caplog.text
        assert "step=weights" in caplog.text
        assert "step=distribution" in caplog.text
        assert "step=escalation" in caplog.text
        assert "step=rationale" in caplog.text
        assert f"thesis={thesis}" in caplog.text
        assert "meta_agent decision ticker=AAPL" in caplog.text
        assert f"recommendation={report.recommendation}" in caplog.text

    async def test_logs_critique_reasoning_when_agents_conflict(
        self, technicals_claim_factory, fundamentals_claim_factory, caplog
    ):
        t = technicals_claim_factory(direction="bullish", confidence=0.8)
        f = fundamentals_claim_factory(direction="bearish", confidence=0.8)
        revised_t = technicals_claim_factory(direction="neutral", confidence=0.5)
        revised_f = fundamentals_claim_factory(direction="neutral", confidence=0.5)
        llm = FakeLLMClient(
            responses=[
                revised_t,
                revised_f,
                _ThesisResponse(thesis="Agents converged after reviewing counter-evidence."),
            ]
        )

        caplog.set_level(logging.INFO)
        await MetaAgent(llm).run("AAPL", 21, t, f)

        assert "step=conflict_check conflict=true" in caplog.text
        assert "step=critique_round" in caplog.text
        assert "pre_technicals=" in caplog.text
        assert "post_technicals=" in caplog.text
        assert "step=post_critique_conflict" in caplog.text
