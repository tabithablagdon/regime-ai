"""Renders a ForecastReport to the Markdown structure in PRD §10.3 — Track D.
Pure function, no I/O.

Structure (§10.3):
  1. Header: ticker, horizon, generated timestamp, recommendation, confidence
  2. Probability distribution table (bullish/neutral/bearish %)
  3. Thesis paragraph
  4. Dissent log: both agents' raw claims side by side, + critique exchange if any
  5. Citations: every filing/news source referenced, with date
  6. Footer disclaimer, printed unconditionally on every report
"""

from __future__ import annotations

from equity_ensemble.schemas.models import AgentClaim, ForecastReport

DISCLAIMER_FOOTER = (
    "Decision support only — not a trade recommendation. "
    "Requires human review before any action."
)

_RECOMMENDATION_LABELS = {
    "bullish_lean": "Bullish Lean",
    "neutral": "Neutral",
    "bearish_lean": "Bearish Lean",
    "requires_review": "Requires Analyst Review",
}

_AGENT_DISPLAY_NAMES = {
    "technicals": "Technicals",
    "fundamentals_sentiment": "Fundamentals/Sentiment",
}


def _render_claim(claim: AgentClaim) -> list[str]:
    lines = [f"### {_AGENT_DISPLAY_NAMES.get(claim.agent, claim.agent)}"]
    lines.append(f"- Direction: {claim.direction}")
    lines.append(f"- Magnitude: {claim.magnitude_bps} bps")
    lines.append(f"- Confidence: {claim.confidence:.0%}")
    if claim.regime_label:
        suffix = " (changed)" if claim.regime_changed else ""
        lines.append(f"- Regime: {claim.regime_label}{suffix}")
    if claim.sentiment_score is not None:
        lines.append(f"- Sentiment score: {claim.sentiment_score}/100")
    if claim.key_risk_flags:
        lines.append(f"- Risk flags: {', '.join(claim.key_risk_flags)}")
    lines.append(f"- Falsifiers: {'; '.join(claim.falsifiers)}")
    lines.append("")
    return lines


def render_markdown(report: ForecastReport) -> str:
    lines: list[str] = []

    # 1. Header
    lines.append(f"# {report.ticker} Forecast — {report.horizon_days}-Day Horizon")
    lines.append("")
    lines.append(f"**Generated:** {report.generated_at.isoformat()}  ")
    recommendation = _RECOMMENDATION_LABELS.get(report.recommendation, report.recommendation)
    lines.append(f"**Recommendation:** {recommendation}  ")
    lines.append(f"**Overall confidence:** {report.overall_confidence:.0%}  ")
    if report.escalate_to_analyst:
        lines.append(f"**Requires Analyst Review:** {report.escalation_reason}  ")
    lines.append("")

    # 2. Probability distribution
    lines.append("## Probability Distribution")
    lines.append("")
    lines.append("| Direction | Probability |")
    lines.append("|---|---|")
    lines.append(f"| Bullish | {report.distribution.bullish_pct:.1f}% |")
    lines.append(f"| Neutral | {report.distribution.neutral_pct:.1f}% |")
    lines.append(f"| Bearish | {report.distribution.bearish_pct:.1f}% |")
    lines.append("")

    # 3. Thesis
    lines.append("## Thesis")
    lines.append("")
    lines.append(report.thesis)
    lines.append("")

    # 4. Dissent log
    lines.append("## Dissent Log")
    lines.append("")
    for claim in report.agent_claims:
        lines.extend(_render_claim(claim))

    if report.critique_log:
        lines.append("**Critique exchange:**")
        lines.append("")
        lines.extend(f"- {entry}" for entry in report.critique_log)
        lines.append("")

    # 5. Citations
    lines.append("## Citations")
    lines.append("")
    if report.citations:
        lines.extend(
            f"- {c.source} ({c.date.isoformat()}): {c.snippet}" for c in report.citations
        )
    else:
        lines.append("- No citations recorded.")
    lines.append("")

    # 6. Footer disclaimer — unconditional
    lines.append("---")
    lines.append("")
    lines.append(DISCLAIMER_FOOTER)

    return "\n".join(lines) + "\n"
