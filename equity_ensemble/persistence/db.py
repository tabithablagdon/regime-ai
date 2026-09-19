"""Persistence interface (PRD §13).

`Database` is the protocol every agent, the Meta-Agent, and the graph code
against. `FakeDatabase` is an in-memory implementation for unit tests so
Tracks A/B/C never need a live Postgres instance. `PostgresDatabase` (Track
E) is the real `asyncpg`-backed implementation, filled in against the same
method signatures the fake already satisfies.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Protocol

import asyncpg

from equity_ensemble.schemas.models import ForecastReport, RetrievalChunk


class Database(Protocol):
    async def get_last_regime(self, ticker: str) -> str | None:
        """Most recent regime_label for this ticker, or None if never
        classified before (Technicals agent long-term memory, PRD §6.1)."""
        ...

    async def save_regime(self, ticker: str, as_of: date, regime_label: str) -> None:
        ...

    async def save_chunks(self, chunks: list[RetrievalChunk]) -> None:
        """Persist filing/news chunks + embeddings (retrieval log, PRD §4.1)."""
        ...

    async def search_chunks(
        self, ticker: str, query_embedding: list[float], limit: int = 10
    ) -> list[RetrievalChunk]:
        """Nearest-neighbor candidates by cosine similarity, most similar
        first. `retrieval.py` (Track B) applies the recency-decay re-ranking
        from PRD §7.2 on top of these candidates."""
        ...

    async def save_forecast(self, report: ForecastReport) -> None:
        """Persist the full run — both raw agent claims, the critique
        exchange if any, and the final synthesis — before the response is
        returned (PRD §4.2, full audit trail requirement)."""
        ...

    async def get_forecast(self, run_id: str) -> ForecastReport | None:
        ...


class FakeDatabase:
    """In-memory `Database` for unit tests."""

    def __init__(self) -> None:
        self._regimes: dict[str, tuple[date, str]] = {}
        self._chunks: list[RetrievalChunk] = []
        self._forecasts: dict[str, ForecastReport] = {}

    async def get_last_regime(self, ticker: str) -> str | None:
        entry = self._regimes.get(ticker.upper())
        return entry[1] if entry else None

    async def save_regime(self, ticker: str, as_of: date, regime_label: str) -> None:
        self._regimes[ticker.upper()] = (as_of, regime_label)

    async def save_chunks(self, chunks: list[RetrievalChunk]) -> None:
        self._chunks.extend(chunks)

    async def search_chunks(
        self, ticker: str, query_embedding: list[float], limit: int = 10
    ) -> list[RetrievalChunk]:
        # No real vector math in the fake — return the ticker's chunks
        # unordered so ranking-logic tests exercise `retrieval.py`'s own
        # sort, not this stand-in.
        candidates = [c for c in self._chunks if c.ticker.upper() == ticker.upper()]
        return candidates[:limit]

    async def save_forecast(self, report: ForecastReport) -> None:
        self._forecasts[str(report.run_id)] = report

    async def get_forecast(self, run_id: str) -> ForecastReport | None:
        return self._forecasts.get(run_id)


class PostgresDatabase:
    """Real implementation (Track E). Construct via `PostgresDatabase.connect()`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str | None = None) -> PostgresDatabase:
        pool = await asyncpg.create_pool(dsn or os.environ["DATABASE_URL"])
        return cls(pool)

    async def get_last_regime(self, ticker: str) -> str | None:
        raise NotImplementedError("Track E: query regime_history, most recent row for ticker")

    async def save_regime(self, ticker: str, as_of: date, regime_label: str) -> None:
        raise NotImplementedError("Track E: insert into regime_history")

    async def save_chunks(self, chunks: list[RetrievalChunk]) -> None:
        raise NotImplementedError("Track E: insert into filing_chunks (pgvector column)")

    async def search_chunks(
        self, ticker: str, query_embedding: list[float], limit: int = 10
    ) -> list[RetrievalChunk]:
        raise NotImplementedError("Track E: ORDER BY embedding <=> $1 LIMIT $2 on filing_chunks")

    async def save_forecast(self, report: ForecastReport) -> None:
        raise NotImplementedError("Track E: insert into forecasts (full audit trail, PRD §4.2)")

    async def get_forecast(self, run_id: str) -> ForecastReport | None:
        raise NotImplementedError("Track E: select from forecasts by run_id")
