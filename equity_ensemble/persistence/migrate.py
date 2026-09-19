"""Applies numbered SQL files in persistence/migrations/ against
DATABASE_URL, tracking what's already run in a schema_migrations table.
Plain SQL, not Alembic — MVP scope doesn't need schema-diffing (PRD §13).

Usage: uv run python -m equity_ensemble.persistence.migrate
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def apply_migrations(dsn: str | None = None) -> list[str]:
    conn = await asyncpg.connect(dsn or os.environ["DATABASE_URL"])
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied_rows = await conn.fetch("SELECT filename FROM schema_migrations")
        applied = {row["filename"] for row in applied_rows}

        newly_applied = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
                )
            newly_applied.append(path.name)
        return newly_applied
    finally:
        await conn.close()


def main() -> None:
    applied = asyncio.run(apply_migrations())
    if applied:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("Up to date.")


if __name__ == "__main__":
    main()
