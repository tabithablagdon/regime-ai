from __future__ import annotations

import os

import asyncpg
import pytest

from equity_ensemble.persistence.db import PostgresDatabase

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://equity_ensemble:equity_ensemble@localhost:5433/equity_ensemble",
)


@pytest.fixture
async def postgres_db():
    """Real `PostgresDatabase` against a local dev instance (see
    docker-compose.yml + `uv run python -m equity_ensemble.persistence.migrate`).

    Skips rather than fails when no Postgres is reachable, so the rest of
    the suite (which never needs a live DB) stays green in any environment.
    """
    try:
        conn = await asyncpg.connect(TEST_DATABASE_URL, timeout=2)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"no Postgres reachable at {TEST_DATABASE_URL}: {exc}")

    try:
        await conn.execute("TRUNCATE forecasts, filing_chunks, regime_history")
    finally:
        await conn.close()

    db = await PostgresDatabase.connect(TEST_DATABASE_URL)
    try:
        yield db
    finally:
        await db.close()
