"""Meta-Agent — Track C (PRD §6.3, §8).

Deterministic pipeline, not a search tree: validate (upstream, via
agents.base) -> conflict-check -> (conditional) one critique round -> fit
distribution -> generate rationale.

Built and tested against hand-written AgentClaim fixtures — does not
require Track A or Track B to be finished.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from pydantic import BaseModel

from equity_ensemble.agents.trace import log_event, summarize_claim
from equity_ensemble.llm.client import LLMClient
from equity_ensemble.schemas.models import AgentClaim, Distribution, Evidence, ForecastReport

logger = logging.getLogger(__name__)

CONFLICT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_REGIME_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "config" / "regime_weights.yaml"

# Distribution fitting (§8.5)
NEUTRAL_BAND_BPS = 50.0
MIN_STD_BPS = 50.0
MAX_STD_BPS = 500.0
DEFAULT_N_SAMPLES = 200_000

# Escalation (§8.1, §8.3)
DEFAULT_CONFIDENCE_ESCALATION_THRESHOLD = 0.4

_DIRECTION_SIGN = {"bullish": 1, "neutral": 0, "bearish": -1}

RATIONALE_SYSTEM_PROMPT = (
    "You are the Meta-Agent for a two-agent equity forecast ensemble. Write a "
    "concise, decision-support thesis paragraph (3-5 sentences). Reference only "
    "the computed probability distribution and the agent claims already provided "
    "below — the numeric fields are already fixed; never invent your own numbers "
    "or restate a different distribution. Never use the words 'buy' or 'sell' or "
    "otherwise recommend a trade; this is decision support only."
)

CRITIQUE_SYSTEM_PROMPT = (
    "You are one of two specialist agents in an equity forecast ensemble. The "
    "other specialist's claim materially disagrees with yours. Review its "
    "evidence below, then return your own claim again in the same schema — "
    "revise your confidence and magnitude_bps if the counter-evidence changes "
    "your view, or keep them if it doesn't. Do not change your `agent` or "
    "`ticker` fields."
)


class _ThesisResponse(BaseModel):
    thesis: str


def check_conflict(technicals: AgentClaim, fundamentals: AgentClaim) -> bool:
    """Material disagreement (PRD §8.2): opposed directions (bullish vs.
    bearish, not neutral vs. either) AND both confidence > 0.5. A simple,
    auditable rule — not learned or LLM-judged."""
    opposed = {technicals.direction, fundamentals.direction} == {"bullish", "bearish"}
    both_confident = (
        technicals.confidence > CONFLICT_CONFIDENCE_THRESHOLD
        and fundamentals.confidence > CONFLICT_CONFIDENCE_THRESHOLD
    )
    return opposed and both_confident


def _conflict_reason(technicals: AgentClaim, fundamentals: AgentClaim) -> str:
    opposed = {technicals.direction, fundamentals.direction} == {"bullish", "bearish"}
    both_confident = (
        technicals.confidence > CONFLICT_CONFIDENCE_THRESHOLD
        and fundamentals.confidence > CONFLICT_CONFIDENCE_THRESHOLD
    )
    if opposed and both_confident:
        return (
            f"opposed directions ({technicals.direction} vs {fundamentals.direction}) "
            f"and both confidences above {CONFLICT_CONFIDENCE_THRESHOLD}"
        )
    if opposed:
        return (
            f"directions oppose ({technicals.direction} vs {fundamentals.direction}) "
            f"but at least one confidence is at or below {CONFLICT_CONFIDENCE_THRESHOLD}"
        )
    return (
        f"directions do not oppose "
        f"({technicals.direction} vs {fundamentals.direction})"
    )


def _critique_prompt(*, own: AgentClaim, other: AgentClaim) -> str:
    other_evidence = "\n".join(f"- {e.source} ({e.date}): {e.snippet}" for e in other.evidence)
    return (
        f"Your prior claim: direction={own.direction}, magnitude_bps={own.magnitude_bps}, "
        f"confidence={own.confidence}.\n\n"
        f"The other specialist ({other.agent}) claims direction={other.direction}, "
        f"magnitude_bps={other.magnitude_bps}, confidence={other.confidence}, citing:\n"
        f"{other_evidence}\n\n"
        "Reconsider your claim in light of this counter-evidence and return your revised claim."
    )


def _preserve_identity_and_regime(revised: AgentClaim, original: AgentClaim) -> AgentClaim:
    """The critique round may revise direction, magnitude, and confidence
    (PRD §8.3). It may not reassign which agent is speaking, or restate the
    regime: that is classified in code by the Technicals agent (PRD §6.1),
    and a claim coming back through the LLM is exactly where a fabricated
    label would slip in. Enforced here rather than requested in the prompt,
    matching how `TechnicalsAgent.run` pins the same fields.

    Without this the revised claims came back with invented labels (observed
    live: `breakdown` for Technicals, whose classifier had said
    `Mean-Reverting`, and `bull_market` for Fundamentals/Sentiment, which has
    no regime of its own), and that label reached the rationale prompt and
    then the user-facing thesis.
    """
    return revised.model_copy(
        update={
            "agent": original.agent,
            "ticker": original.ticker,
            "regime_label": original.regime_label,
            "regime_changed": original.regime_changed,
        }
    )


async def run_critique_round(
    llm: LLMClient, technicals: AgentClaim, fundamentals: AgentClaim
) -> tuple[AgentClaim, AgentClaim, list[str]]:
    """Single exchange (PRD §8.3): each specialist sees the other's claim and
    evidence, reconsiders confidence/magnitude. Nothing is overwritten —
    caller keeps the pre-critique originals for the dissent log; this
    returns the revised claims plus a human-readable log of what changed."""
    revised_technicals = _preserve_identity_and_regime(
        await llm.complete_structured(
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            user_prompt=_critique_prompt(own=technicals, other=fundamentals),
            response_model=AgentClaim,
        ),
        technicals,
    )
    revised_fundamentals = _preserve_identity_and_regime(
        await llm.complete_structured(
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            user_prompt=_critique_prompt(own=fundamentals, other=technicals),
            response_model=AgentClaim,
        ),
        fundamentals,
    )

    critique_log = [
        f"[pre-critique] technicals: {technicals.direction} "
        f"(confidence={technicals.confidence}, magnitude_bps={technicals.magnitude_bps})",
        f"[pre-critique] fundamentals_sentiment: {fundamentals.direction} "
        f"(confidence={fundamentals.confidence}, magnitude_bps={fundamentals.magnitude_bps})",
        f"[post-critique] technicals: {revised_technicals.direction} "
        f"(confidence={revised_technicals.confidence}, "
        f"magnitude_bps={revised_technicals.magnitude_bps})",
        f"[post-critique] fundamentals_sentiment: {revised_fundamentals.direction} "
        f"(confidence={revised_fundamentals.confidence}, "
        f"magnitude_bps={revised_fundamentals.magnitude_bps})",
    ]
    log_event(
        logger,
        "meta_agent",
        "thinking",
        ticker=technicals.ticker,
        step="critique_round",
        pre_technicals=summarize_claim(technicals),
        pre_fundamentals=summarize_claim(fundamentals),
        post_technicals=summarize_claim(revised_technicals),
        post_fundamentals=summarize_claim(revised_fundamentals),
    )
    return revised_technicals, revised_fundamentals, critique_log


@lru_cache(maxsize=8)
def _load_weights_table(weights_path: str) -> dict[str, dict[str, float]]:
    with open(weights_path) as f:
        return yaml.safe_load(f)


def resolve_weights(
    regime_label: str | None, *, weights_path: Path = DEFAULT_REGIME_WEIGHTS_PATH
) -> dict[str, float]:
    """Regime-conditioned static lookup (PRD §8.4). Falls back to the
    `default` entry if `regime_label` is missing or not in the table."""
    table = _load_weights_table(str(weights_path))
    if regime_label and regime_label in table:
        return table[regime_label]
    return table["default"]


def _claim_to_normal(claim: AgentClaim) -> tuple[float, float]:
    mean = _DIRECTION_SIGN[claim.direction] * claim.magnitude_bps
    std = MIN_STD_BPS + (1 - claim.confidence) * (MAX_STD_BPS - MIN_STD_BPS)
    return mean, std


def fit_distribution(
    technicals: AgentClaim | None,
    fundamentals: AgentClaim | None,
    weights: dict[str, float],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> Distribution:
    """Convert each available claim to normal(mean=signed magnitude,
    std=f(1-confidence)), combine into a weighted mixture via Monte Carlo
    sampling, summarize into the 3-bucket distribution (PRD §8.5). No LLM
    ever performs this arithmetic."""
    claims = [
        (name, claim)
        for name, claim in (("technicals", technicals), ("fundamentals_sentiment", fundamentals))
        if claim is not None
    ]
    if not claims:
        raise ValueError("at least one claim is required to fit a distribution")

    total_weight = sum(weights.get(name, 0.0) for name, _ in claims)
    if total_weight <= 0:
        normalized = [1.0 / len(claims)] * len(claims)
    else:
        normalized = [weights.get(name, 0.0) / total_weight for name, _ in claims]

    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    offset = 0
    for i, ((_, claim), w) in enumerate(zip(claims, normalized, strict=True)):
        mean, std = _claim_to_normal(claim)
        count = n_samples - offset if i == len(claims) - 1 else round(w * n_samples)
        samples[offset : offset + count] = rng.normal(mean, std, count)
        offset += count

    bullish_count = int(np.sum(samples > NEUTRAL_BAND_BPS))
    bearish_count = int(np.sum(samples < -NEUTRAL_BAND_BPS))
    neutral_count = n_samples - bullish_count - bearish_count

    return Distribution(
        bullish_pct=bullish_count / n_samples * 100,
        neutral_pct=neutral_count / n_samples * 100,
        bearish_pct=bearish_count / n_samples * 100,
    )


def _dedupe_citations(claims: list[AgentClaim]) -> list[Evidence]:
    seen: set[tuple[str, str]] = set()
    citations: list[Evidence] = []
    for claim in claims:
        for evidence in claim.evidence:
            key = (evidence.source, evidence.date.isoformat())
            if key not in seen:
                seen.add(key)
                citations.append(evidence)
    return citations


def _recommendation(distribution: Distribution, *, escalate: bool) -> str:
    if escalate:
        return "requires_review"
    buckets = {
        "bullish_lean": distribution.bullish_pct,
        "neutral": distribution.neutral_pct,
        "bearish_lean": distribution.bearish_pct,
    }
    return max(buckets, key=buckets.get)


async def generate_rationale(
    llm: LLMClient,
    *,
    ticker: str,
    distribution: Distribution,
    technicals: AgentClaim | None,
    fundamentals: AgentClaim | None,
    critique_log: list[str],
) -> str:
    """LLM call happens last, after every numeric field is fixed (PRD §8.6)
    — the model narrates the already-computed result, it never invents it."""
    lines = [
        f"Ticker: {ticker}",
        f"Computed distribution: bullish={distribution.bullish_pct:.1f}%, "
        f"neutral={distribution.neutral_pct:.1f}%, bearish={distribution.bearish_pct:.1f}%",
    ]
    if technicals is not None:
        lines.append(
            f"Technicals claim: {technicals.direction}, confidence={technicals.confidence}, "
            f"regime={technicals.regime_label}"
        )
    if fundamentals is not None:
        lines.append(
            f"Fundamentals/Sentiment claim: {fundamentals.direction}, "
            f"confidence={fundamentals.confidence}, sentiment_score={fundamentals.sentiment_score}"
        )
    if critique_log:
        lines.append("Critique exchange:\n" + "\n".join(critique_log))

    response = await llm.complete_structured(
        system_prompt=RATIONALE_SYSTEM_PROMPT,
        user_prompt="\n".join(lines),
        response_model=_ThesisResponse,
    )
    log_event(
        logger,
        "meta_agent",
        "thinking",
        ticker=ticker,
        step="rationale",
        thesis=response.thesis,
    )
    return response.thesis


class MetaAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        confidence_escalation_threshold: float = DEFAULT_CONFIDENCE_ESCALATION_THRESHOLD,
    ) -> None:
        self.llm = llm
        self.confidence_escalation_threshold = confidence_escalation_threshold

    async def run(
        self,
        ticker: str,
        horizon_days: int,
        technicals: AgentClaim | None,
        fundamentals: AgentClaim | None,
    ) -> ForecastReport:
        """Full synthesis pipeline (PRD §8). `technicals`/`fundamentals` may
        be None if that agent was marked `AgentUnavailable` upstream — the
        caller (graph join node) is responsible for that retry/escalation
        decision (PRD §8.1); this method just reports it."""
        original_claims = [c for c in (technicals, fundamentals) if c is not None]
        if not original_claims:
            raise ValueError("MetaAgent.run requires at least one available agent claim")

        log_event(
            logger,
            "meta_agent",
            "called",
            ticker=ticker,
            horizon_days=horizon_days,
            technicals=summarize_claim(technicals),
            fundamentals=summarize_claim(fundamentals),
        )

        critique_log: list[str] = []
        escalation_reasons: list[str] = []
        fitting_technicals, fitting_fundamentals = technicals, fundamentals

        if technicals is None or fundamentals is None:
            missing = "technicals" if technicals is None else "fundamentals_sentiment"
            reason = f"{missing} claim unavailable; report generated in single-agent mode"
            escalation_reasons.append(reason)
            log_event(
                logger,
                "meta_agent",
                "thinking",
                ticker=ticker,
                step="conflict_check",
                conflict=False,
                reason=reason,
            )
        else:
            conflict = check_conflict(technicals, fundamentals)
            reason = _conflict_reason(technicals, fundamentals)
            log_event(
                logger,
                "meta_agent",
                "thinking",
                ticker=ticker,
                step="conflict_check",
                conflict=conflict,
                reason=reason,
            )
            if conflict:
                fitting_technicals, fitting_fundamentals, critique_log = await run_critique_round(
                    self.llm, technicals, fundamentals
                )
                still_conflict = check_conflict(fitting_technicals, fitting_fundamentals)
                if still_conflict:
                    escalation_reasons.append(
                        "agents materially disagree even after the single critique round"
                    )
                log_event(
                    logger,
                    "meta_agent",
                    "thinking",
                    ticker=ticker,
                    step="post_critique_conflict",
                    conflict=still_conflict,
                    reason=_conflict_reason(fitting_technicals, fitting_fundamentals),
                )

        regime_label = technicals.regime_label if technicals is not None else None
        weights = resolve_weights(regime_label)
        log_event(
            logger,
            "meta_agent",
            "thinking",
            ticker=ticker,
            step="weights",
            regime=regime_label,
            weights=weights,
        )
        distribution = fit_distribution(fitting_technicals, fitting_fundamentals, weights)
        log_event(
            logger,
            "meta_agent",
            "thinking",
            ticker=ticker,
            step="distribution",
            bullish_pct=distribution.bullish_pct,
            neutral_pct=distribution.neutral_pct,
            bearish_pct=distribution.bearish_pct,
            fitting_technicals=summarize_claim(fitting_technicals),
            fitting_fundamentals=summarize_claim(fitting_fundamentals),
        )

        fitting_claims = [c for c in (fitting_technicals, fitting_fundamentals) if c is not None]
        overall_confidence = sum(c.confidence for c in fitting_claims) / len(fitting_claims)
        if overall_confidence < self.confidence_escalation_threshold:
            escalation_reasons.append(
                f"overall confidence {overall_confidence:.2f} below threshold "
                f"{self.confidence_escalation_threshold}"
            )

        escalate = bool(escalation_reasons)
        recommendation = _recommendation(distribution, escalate=escalate)
        log_event(
            logger,
            "meta_agent",
            "thinking",
            ticker=ticker,
            step="escalation",
            escalate=escalate,
            overall_confidence=overall_confidence,
            threshold=self.confidence_escalation_threshold,
            reasons=escalation_reasons or ["none"],
            recommendation=recommendation,
        )

        thesis = await generate_rationale(
            self.llm,
            ticker=ticker,
            distribution=distribution,
            technicals=fitting_technicals,
            fundamentals=fitting_fundamentals,
            critique_log=critique_log,
        )

        report = ForecastReport(
            run_id=uuid4(),
            ticker=ticker,
            horizon_days=horizon_days,
            generated_at=datetime.now(UTC),
            distribution=distribution,
            recommendation=recommendation,
            overall_confidence=overall_confidence,
            escalate_to_analyst=escalate,
            escalation_reason="; ".join(escalation_reasons) if escalation_reasons else None,
            thesis=thesis,
            agent_claims=original_claims,
            critique_log=critique_log,
            citations=_dedupe_citations(original_claims),
        )
        log_event(
            logger,
            "meta_agent",
            "decision",
            ticker=ticker,
            run_id=report.run_id,
            recommendation=report.recommendation,
            overall_confidence=report.overall_confidence,
            escalate_to_analyst=report.escalate_to_analyst,
            escalation_reason=report.escalation_reason,
            distribution=(
                f"bullish={distribution.bullish_pct:.1f},"
                f"neutral={distribution.neutral_pct:.1f},"
                f"bearish={distribution.bearish_pct:.1f}"
            ),
            thesis=report.thesis,
        )
        return report
