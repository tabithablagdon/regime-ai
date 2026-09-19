"""Meta-Agent — owned by Track C (PRD §6.3, §8).

Deterministic pipeline, not a search tree: validate -> conflict-check ->
(conditional) one critique round -> fit distribution -> generate rationale.

Work items (see implementation plan, Track C):
1. check_conflict (§8.2): pure function.
2. run_critique_round (§8.3): one exchange, captures revised claims into
   critique_log without overwriting the originals.
3. load_regime_weights / resolve_weights (§8.4): read config/regime_weights.yaml.
4. fit_distribution (§8.5): NumPy/SciPy, no LLM arithmetic.
5. generate_rationale (§8.6): LLM call last, after all numeric fields are fixed.
6. MetaAgent.run: wires 1-5 into the ForecastReport (§7.3), including
   escalate_to_analyst logic.

Built and tested against hand-written AgentClaim fixtures (agreeing pair,
opposed pair, low-confidence pair, one-agent-unavailable case) — does not
require Track A or Track B to be finished.
"""

from __future__ import annotations

from pathlib import Path

from equity_ensemble.llm.client import LLMClient
from equity_ensemble.schemas.models import AgentClaim, Distribution, ForecastReport

CONFLICT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_REGIME_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "config" / "regime_weights.yaml"


def check_conflict(technicals: AgentClaim, fundamentals: AgentClaim) -> bool:
    """Material disagreement (PRD §8.2): opposed directions (bullish vs.
    bearish, not neutral vs. either) AND both confidence > 0.5."""
    raise NotImplementedError("Track C: implement the conflict rule")


async def run_critique_round(
    llm: LLMClient, technicals: AgentClaim, fundamentals: AgentClaim
) -> tuple[AgentClaim, AgentClaim, list[str]]:
    """Single exchange (PRD §8.3): each specialist sees the other's claim
    and evidence, reconsiders confidence/magnitude. Returns
    (revised_technicals, revised_fundamentals, critique_log)."""
    raise NotImplementedError("Track C: implement the single-round critique")


def resolve_weights(
    regime_label: str | None, *, weights_path: Path = DEFAULT_REGIME_WEIGHTS_PATH
) -> dict[str, float]:
    """Regime-conditioned static lookup (PRD §8.4)."""
    raise NotImplementedError("Track C: load and look up config/regime_weights.yaml")


def fit_distribution(
    technicals: AgentClaim | None, fundamentals: AgentClaim | None, weights: dict[str, float]
) -> Distribution:
    """Convert each available claim to normal(mean=signed magnitude,
    std=f(1-confidence)), combine into a weighted mixture, summarize into
    the 3-bucket distribution (PRD §8.5). No LLM ever performs this math."""
    raise NotImplementedError("Track C: implement NumPy/SciPy distribution fitting")


class MetaAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def run(
        self,
        ticker: str,
        horizon_days: int,
        technicals: AgentClaim | None,
        fundamentals: AgentClaim | None,
    ) -> ForecastReport:
        """Full synthesis pipeline (PRD §8): validate (handled upstream by
        agents.base) -> check_conflict -> run_critique_round if needed ->
        resolve_weights -> fit_distribution -> generate_rationale ->
        assemble ForecastReport with escalate_to_analyst set per PRD §8.1/§8.3."""
        raise NotImplementedError("Track C: wire the pipeline described above")
