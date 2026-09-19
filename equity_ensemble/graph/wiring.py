"""Builds the production dependency stack (real OpenRouter/FMP/Postgres/
Voyage) so the API and CLI construct an identical graph — Track D. Requires
the env vars in .env.example to be set; nothing here is exercised by the
test suite, which builds graphs from stub/fake dependencies directly.
"""

from __future__ import annotations

from equity_ensemble.agents.fundamentals_sentiment import FundamentalsSentimentAgent
from equity_ensemble.agents.meta_agent import MetaAgent
from equity_ensemble.agents.technicals import TechnicalsAgent
from equity_ensemble.data.fmp_client import FMPClient, RealFMPClient
from equity_ensemble.data.retrieval import VoyageEmbedder
from equity_ensemble.graph.build_graph import build_graph
from equity_ensemble.llm.client import OpenRouterClient
from equity_ensemble.persistence.db import PostgresDatabase


async def build_production_graph() -> tuple[object, PostgresDatabase, FMPClient]:
    """Returns `(compiled_graph, db, fmp)`. The caller owns `db`'s lifecycle
    (call `await db.close()` on shutdown) and can reuse `fmp` for the
    pre-flight ticker-existence check (PRD §4.2 step 2)."""
    llm = OpenRouterClient()
    fmp = RealFMPClient()
    db = await PostgresDatabase.connect()
    embedder = VoyageEmbedder()

    technicals_agent = TechnicalsAgent(llm, fmp, db)
    fundamentals_agent = FundamentalsSentimentAgent(llm, fmp, db, embedder)
    meta_agent = MetaAgent(llm)

    graph = build_graph(
        technicals_agent=technicals_agent.run,
        fundamentals_agent=fundamentals_agent.run,
        meta_agent=meta_agent.run,
    )
    return graph, db, fmp
