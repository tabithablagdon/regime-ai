"""LangGraph wiring — Track D (PRD §4.1, §4.2).

Fan-out (`START` -> both specialist nodes, which LangGraph runs
concurrently since they share a source) -> both feed a single `synthesize`
node -> `END`.

Deviation from the PRD's literal §4.1 topology, noted for transparency: the
PRD describes a graph-level "conditional edge for the critique loop." Track
C (agents/meta_agent.py) already implements the conflict-check/critique/
distribution/rationale pipeline as a single deterministic method
(`MetaAgent.run`), which is the MVP simplification the PRD itself asks for
(§2.2: replace ToT/beam search with a deterministic pipeline). Re-exploding
that into separate graph nodes would just relocate the same logic without
changing behavior, so `synthesize` calls it as one node rather than as
several conditionally-wired ones.

If a specialist raises `AgentUnavailable` (PRD §8.1, two failed validation
retries), its node stores `None` rather than crashing the run — `synthesize`
already handles a `None` claim as single-agent mode. If *both* specialists
are unavailable, `MetaAgent.run` raises `ValueError` and that propagates out
of the compiled graph; the API/CLI layer is responsible for turning that
into a 5xx / error message, since the PRD only specifies graceful
degradation for a single failed agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from equity_ensemble.agents.base import AgentUnavailable
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
    run_id: str = ""
    technicals_claim: AgentClaim | None = None
    fundamentals_claim: AgentClaim | None = None
    report: ForecastReport | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = str(uuid4())


def build_graph(
    *,
    technicals_agent: SpecialistCallable,
    fundamentals_agent: SpecialistCallable,
    meta_agent: MetaAgentCallable,
) -> Any:
    """Returns a compiled LangGraph `StateGraph` over `GraphState`. Pass
    bound agent methods (e.g. `TechnicalsAgent(...).run`) as the callables —
    each just needs to match the Protocol signatures above."""

    async def technicals_node(state: GraphState) -> dict:
        try:
            claim = await technicals_agent(state.ticker, state.horizon_days)
        except AgentUnavailable:
            claim = None
        return {"technicals_claim": claim}

    async def fundamentals_node(state: GraphState) -> dict:
        try:
            claim = await fundamentals_agent(state.ticker, state.horizon_days)
        except AgentUnavailable:
            claim = None
        return {"fundamentals_claim": claim}

    async def synthesize_node(state: GraphState) -> dict:
        report = await meta_agent(
            state.ticker, state.horizon_days, state.technicals_claim, state.fundamentals_claim
        )
        return {"report": report}

    graph = StateGraph(GraphState)
    graph.add_node("technicals", technicals_node)
    graph.add_node("fundamentals", fundamentals_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "technicals")
    graph.add_edge(START, "fundamentals")
    graph.add_edge("technicals", "synthesize")
    graph.add_edge("fundamentals", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_forecast(
    compiled_graph: Any, *, ticker: str, horizon_days: int, run_id: str | None = None
) -> ForecastReport:
    """Convenience wrapper used by both api/main.py and cli/main.py so
    their behavior can never diverge (PRD's own stated goal for the
    CLI/API split)."""
    initial_state = GraphState(ticker=ticker, horizon_days=horizon_days, run_id=run_id or "")
    final_state = await compiled_graph.ainvoke(initial_state)
    report = final_state["report"] if isinstance(final_state, dict) else final_state.report
    if report is None:
        raise RuntimeError("graph completed without producing a ForecastReport")
    return report
