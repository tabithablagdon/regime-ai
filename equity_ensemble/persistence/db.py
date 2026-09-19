"""Persistence interface (PRD §13).

`Database` is the protocol every agent, the Meta-Agent, and the graph code
against. `FakeDatabase` is an in-memory implementation for unit tests so
Tracks A/B/C never need a live Postgres instance. `PostgresDatabase` (Track
E) is the real `asyncpg`-backed implementation, against the same method
signatures the fake already satisfies.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta
from typing import Protocol

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

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

    async def list_pending_evaluations(self, *, as_of: date | None = None) -> list[ForecastReport]:
        """Forecasts whose horizon has elapsed (approximated as calendar
        days for MVP) with no realized outcome recorded yet (PRD §12)."""
        ...

    async def record_realized_outcome(
        self, run_id: str, *, realized_return_bps: float, brier_score: float
    ) -> None:
        """Writes the realized return and Brier score back onto a forecast
        once its horizon has elapsed (PRD §12, eval/brier_score.py)."""
        ...


class FakeDatabase:
    """In-memory `Database` for unit tests."""

    def __init__(self) -> None:
        self._regimes: dict[str, tuple[date, str]] = {}
        self._chunks: list[RetrievalChunk] = []
        self._forecasts: dict[str, ForecastReport] = {}
        self._outcomes: dict[str, dict[str, float]] = {}

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

    async def list_pending_evaluations(self, *, as_of: date | None = None) -> list[ForecastReport]:
        as_of = as_of or date.today()
        pending = []
        for run_id, report in self._forecasts.items():
            if run_id in self._outcomes:
                continue
            horizon_elapsed = as_of - report.generated_at.date() >= timedelta(
                days=report.horizon_days
            )
            if horizon_elapsed:
                pending.append(report)
        return pending

    async def record_realized_outcome(
        self, run_id: str, *, realized_return_bps: float, brier_score: float
    ) -> None:
        self._outcomes[run_id] = {
            "realized_return_bps": realized_return_bps,
            "brier_score": brier_score,
        }


class PostgresDatabase:
    """Real implementation (Track E). Construct via `PostgresDatabase.connect()`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str | None = None) -> PostgresDatabase:
        pool = await asyncpg.create_pool(
            dsn or os.environ["DATABASE_URL"], init=register_vector
        )
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def get_last_regime(self, ticker: str) -> str | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT regime_label FROM regime_history WHERE ticker = $1 "
                "ORDER BY as_of DESC LIMIT 1",
                ticker.upper(),
            )
        return row["regime_label"] if row else None

    async def save_regime(self, ticker: str, as_of: date, regime_label: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO regime_history (ticker, as_of, regime_label) VALUES ($1, $2, $3)",
                ticker.upper(),
                as_of,
                regime_label,
            )

    async def save_chunks(self, chunks: list[RetrievalChunk]) -> None:
        if not chunks:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO filing_chunks
                    (chunk_id, ticker, source_type, source_date, section, text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                [
                    (
                        c.chunk_id,
                        c.ticker.upper(),
                        c.source_type,
                        c.source_date,
                        c.section,
                        c.text,
                        np.asarray(c.embedding, dtype=np.float32),
                    )
                    for c in chunks
                ],
            )

    async def search_chunks(
        self, ticker: str, query_embedding: list[float], limit: int = 10
    ) -> list[RetrievalChunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_id, ticker, source_type, source_date, section, text, embedding
                FROM filing_chunks
                WHERE ticker = $1
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                ticker.upper(),
                np.asarray(query_embedding, dtype=np.float32),
                limit,
            )
        return [
            RetrievalChunk(
                chunk_id=row["chunk_id"],
                ticker=row["ticker"],
                source_type=row["source_type"],
                source_date=row["source_date"],
                section=row["section"],
                text=row["text"],
                embedding=row["embedding"].to_list(),
            )
            for row in rows
        ]

    async def save_forecast(self, report: ForecastReport) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO forecasts (run_id, ticker, horizon_days, generated_at, report)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (run_id) DO NOTHING
                """,
                report.run_id,
                report.ticker,
                report.horizon_days,
                report.generated_at,
                report.model_dump_json(),
            )

    async def get_forecast(self, run_id: str) -> ForecastReport | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT report FROM forecasts WHERE run_id = $1", uuid.UUID(str(run_id))
            )
        return ForecastReport.model_validate_json(row["report"]) if row else None

    async def list_pending_evaluations(self, *, as_of: date | None = None) -> list[ForecastReport]:
        as_of = as_of or date.today()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT report FROM forecasts
                WHERE realized_return IS NULL
                  AND generated_at + (horizon_days || ' days')::interval <= $1
                """,
                datetime.combine(as_of, datetime.min.time()),
            )
        return [ForecastReport.model_validate_json(row["report"]) for row in rows]

    async def record_realized_outcome(
        self, run_id: str, *, realized_return_bps: float, brier_score: float
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE forecasts
                SET realized_return = $2, brier_score = $3
                WHERE run_id = $1
                """,
                uuid.UUID(str(run_id)),
                realized_return_bps,
                brier_score,
            )
