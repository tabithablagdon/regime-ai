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

If a specialist fails, its node stores `None` rather than crashing the run —
`synthesize` already handles a `None` claim as single-agent mode. That covers
both failure classes PRD §8.1 and §11 call out: `AgentUnavailable` (two
failed schema-validation retries) and a vendor failure such as a rate limit
or an outage. If *both* specialists are unavailable, `MetaAgent.run` raises
`ValueError` and that propagates out of the compiled graph; the API/CLI layer
is responsible for turning that into a 5xx / error message, since the PRD
only specifies graceful degradation for a single failed agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from equity_ensemble.agents.base import AgentUnavailable
from equity_ensemble.agents.trace import log_event
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim, ForecastReport

logger = logging.getLogger(__name__)

DEFAULT_CACHE_MAX_AGE = timedelta(hours=24)


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

    async def run_specialist(
        agent: SpecialistCallable, name: str, state: GraphState
    ) -> AgentClaim | None:
        """Returns `None` instead of raising, so one failed specialist costs
        its own claim and not the whole run. `AgentUnavailable` is the
        schema-retry path (PRD §8.1); any other exception is a vendor or
        upstream failure, which PRD §11 also requires degrading to
        single-agent mode rather than failing the request. The Meta-Agent
        turns a `None` claim into an escalated single-agent report, so the
        cause is stated in the report instead of being silently dropped.
        """
        try:
            return await agent(state.ticker, state.horizon_days)
        except AgentUnavailable:
            reason = "failed schema validation twice"
        except Exception as exc:
            # Full traceback server-side; the audit line keeps just the type
            # so it stays one readable line.
            logger.exception("%s agent failed for ticker=%s", name, state.ticker)
            reason = type(exc).__name__

        log_event(
            logger,
            name,
            "unavailable",
            ticker=state.ticker,
            horizon_days=state.horizon_days,
            run_id=state.run_id,
            reason=reason,
        )
        return None

    async def technicals_node(state: GraphState) -> dict:
        claim = await run_specialist(technicals_agent, "technicals", state)
        return {"technicals_claim": claim}

    async def fundamentals_node(state: GraphState) -> dict:
        claim = await run_specialist(fundamentals_agent, "fundamentals_sentiment", state)
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


@dataclass
class ForecastResult:
    report: ForecastReport
    cache_hit: bool


async def run_forecast(
    compiled_graph: Any,
    *,
    ticker: str,
    horizon_days: int,
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    db: Database | None = None,
    cache_max_age: timedelta | None = DEFAULT_CACHE_MAX_AGE,
) -> ForecastResult:
    """Convenience wrapper used by both api/main.py and cli/main.py so
    their behavior can never diverge (PRD's own stated goal for the
    CLI/API split). `config` is forwarded as-is to the graph invocation —
    e.g. `{"callbacks": [tracer]}` to attach LangSmith tracing for a single
    request without making it a global default.

    `db` is optional (tests that don't care about persistence just omit
    it, unchanged from before this existed). When given, two things happen
    around the graph invocation rather than inside it:

    1. Before: a forecast for this exact (ticker, horizon_days) generated
       within `cache_max_age` short-circuits the whole pipeline — no LLM,
       FMP, or embedding calls — and that cached report is returned as-is.
       Pass `cache_max_age=None` to persist without ever reading the cache.
    2. After a fresh run: the report is persisted (PRD §4.2's full audit
       trail requirement), which is also what makes the cache above have
       anything to find on a later call.
    """
    if db is not None and cache_max_age is not None:
        cached = await db.get_recent_forecast(ticker, horizon_days, max_age=cache_max_age)
        if cached is not None:
            log_event(
                logger,
                "forecast",
                "cache_hit",
                ticker=ticker,
                run_id=cached.run_id,
                generated_at=cached.generated_at.isoformat(),
            )
            return ForecastResult(report=cached, cache_hit=True)

    initial_state = GraphState(ticker=ticker, horizon_days=horizon_days, run_id=run_id or "")
    log_event(
        logger,
        "forecast",
        "called",
        ticker=ticker,
        horizon_days=horizon_days,
        run_id=initial_state.run_id,
    )
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    report = final_state["report"] if isinstance(final_state, dict) else final_state.report
    if report is None:
        raise RuntimeError("graph completed without producing a ForecastReport")
    log_event(
        logger,
        "forecast",
        "decision",
        ticker=ticker,
        run_id=report.run_id,
        recommendation=report.recommendation,
        overall_confidence=report.overall_confidence,
        escalate_to_analyst=report.escalate_to_analyst,
    )

    if db is not None:
        await db.save_forecast(report)

    return ForecastResult(report=report, cache_hit=False)
