"""LangGraph wiring — owned by Track D (PRD §4.1, §4.2).

Fan-out node -> Technicals node ∥ Fundamentals/Sentiment node (concurrent)
-> join node -> Meta-Agent (validate, conflict-check, conditional critique,
fit, rationale) -> output node.

Build against stub agent callables first (return canned AgentClaim/
ForecastReport instances) — swapping in Track A/B/C's real agents at
integration is a one-line change per node (see implementation plan, Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from equity_ensemble.schemas.models import AgentClaim, ForecastReport


class SpecialistCallable(Protocol):
    async def __call__(self, ticker: str, horizon_days: int) -> AgentClaim: ...


class MetaAgentCallable(Protocol):
    async def __call__(
        self,
        ticker: str,
        horizon_days: int,
        technicals: AgentClaim | None,
        fundamentals: AgentClaim | None,
    ) -> ForecastReport: ...


@dataclass
class GraphState:
    ticker: str
    horizon_days: int
    run_id: str
    technicals_claim: AgentClaim | None = None
    fundamentals_claim: AgentClaim | None = None
    report: ForecastReport | None = None


def build_graph(
    *,
    technicals_agent: SpecialistCallable,
    fundamentals_agent: SpecialistCallable,
    meta_agent: MetaAgentCallable,
) -> Any:
    """Returns a compiled LangGraph `StateGraph` over `GraphState`.

    Track D: implement the fan-out/join/conditional-critique-edge topology
    described in PRD §4.1. Both specialist nodes must run concurrently.
    """
    raise NotImplementedError("Track D: implement the LangGraph StateGraph")
