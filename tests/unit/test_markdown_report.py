from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from equity_ensemble.render.markdown_report import DISCLAIMER_FOOTER, render_markdown
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport


def _claim(**overrides) -> AgentClaim:
    defaults = dict(
        agent="technicals",
        ticker="AAPL",
        direction="bullish",
        magnitude_bps=150,
        confidence=0.7,
        regime_label="Trending Bull",
        regime_changed=True,
        evidence=[Evidence(source="today's OHLCV", date=date(2026, 8, 1), snippet="uptrend")],
        falsifiers=["ADX drops below 20"],
    )
    defaults.update(overrides)
    return AgentClaim(**defaults)


def _report(**overrides) -> ForecastReport:
    defaults = dict(
        run_id=uuid4(),
        ticker="AAPL",
        horizon_days=21,
        generated_at=datetime(2026, 9, 19, 12, 0, 0),
        distribution=Distribution(bullish_pct=60, neutral_pct=25, bearish_pct=15),
        recommendation="bullish_lean",
        overall_confidence=0.65,
        escalate_to_analyst=False,
        escalation_reason=None,
        thesis="Both agents lean bullish given the trending regime.",
        agent_claims=[_claim()],
        critique_log=[],
        citations=[Evidence(source="10-Q Risk Factors", date=date(2026, 8, 1), snippet="...")],
    )
    defaults.update(overrides)
    return ForecastReport(**defaults)


class TestRenderMarkdown:
    def test_footer_disclaimer_always_present(self):
        assert DISCLAIMER_FOOTER in render_markdown(_report())

    def test_footer_disclaimer_present_even_when_escalating(self):
        report = _report(
            escalate_to_analyst=True,
            escalation_reason="agents materially disagree",
            recommendation="requires_review",
        )
        assert DISCLAIMER_FOOTER in render_markdown(report)

    def test_header_contains_ticker_and_horizon(self):
        md = render_markdown(_report())
        assert "AAPL" in md
        assert "21-Day Horizon" in md

    def test_distribution_table_has_all_three_buckets(self):
        md = render_markdown(_report())
        assert "60.0%" in md
        assert "25.0%" in md
        assert "15.0%" in md

    def test_thesis_is_included_verbatim(self):
        md = render_markdown(_report())
        assert "Both agents lean bullish given the trending regime." in md

    def test_dissent_log_includes_each_agent_claim(self):
        technicals = _claim(agent="technicals", direction="bullish")
        fundamentals = _claim(
            agent="fundamentals_sentiment",
            ticker="AAPL",
            direction="bearish",
            sentiment_score=40,
            key_risk_flags=["guidance_cut"],
        )
        md = render_markdown(_report(agent_claims=[technicals, fundamentals]))
        assert "Technicals" in md
        assert "Fundamentals/Sentiment" in md
        assert "guidance_cut" in md
        assert "40/100" in md

    def test_critique_log_shown_when_present(self):
        md = render_markdown(_report(critique_log=["[pre-critique] technicals: bullish"]))
        assert "Critique exchange" in md
        assert "[pre-critique] technicals: bullish" in md

    def test_critique_section_omitted_when_no_critique_ran(self):
        md = render_markdown(_report(critique_log=[]))
        assert "Critique exchange" not in md

    def test_citations_are_listed(self):
        md = render_markdown(_report())
        assert "10-Q Risk Factors" in md

    def test_no_citations_says_so_explicitly(self):
        md = render_markdown(_report(citations=[]))
        assert "No citations recorded." in md

    def test_escalation_reason_shown_when_escalating(self):
        report = _report(
            escalate_to_analyst=True,
            escalation_reason="overall confidence below threshold",
            recommendation="requires_review",
        )
        md = render_markdown(report)
        assert "Requires Analyst Review" in md
        assert "overall confidence below threshold" in md
