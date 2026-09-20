"""Real Postgres+pgvector integration tests for Track E (PRD milestone M6
DoD: every field needed to reconstruct a past report is queryable after the
fact). Requires `docker compose up -d` +
`uv run python -m equity_ensemble.persistence.migrate`; skips automatically
if no instance is reachable (see conftest.py)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from uuid import uuid4

from equity_ensemble.schemas.models import (
    AgentClaim,
    Distribution,
    Evidence,
    ForecastReport,
    RetrievalChunk,
)


def _vector(dim: int, active: dict[int, float]) -> list[float]:
    v = [0.0] * dim
    for i, val in active.items():
        v[i] = val
    return v


class TestRegimeHistory:
    async def test_no_prior_regime_is_none(self, postgres_db):
        assert await postgres_db.get_last_regime("AAPL") is None

    async def test_most_recent_regime_wins(self, postgres_db):
        await postgres_db.save_regime("AAPL", date(2026, 1, 1), "Mean-Reverting")
        await postgres_db.save_regime("AAPL", date(2026, 6, 1), "Trending Bull")

        assert await postgres_db.get_last_regime("AAPL") == "Trending Bull"

    async def test_regime_is_scoped_to_ticker(self, postgres_db):
        await postgres_db.save_regime("AAPL", date(2026, 1, 1), "Trending Bull")
        await postgres_db.save_regime("MSFT", date(2026, 1, 1), "Volatile/Choppy")

        assert await postgres_db.get_last_regime("MSFT") == "Volatile/Choppy"


class TestFilingChunks:
    async def test_search_orders_by_cosine_similarity(self, postgres_db):
        dim = 1024
        query = _vector(dim, {0: 1.0})
        close = RetrievalChunk(
            chunk_id="close-1",
            ticker="AAPL",
            source_type="10-K",
            source_date=date(2026, 1, 1),
            section="Risk Factors",
            text="closely related chunk",
            embedding=_vector(dim, {0: 0.95, 1: 0.05}),
        )
        far = RetrievalChunk(
            chunk_id="far-1",
            ticker="AAPL",
            source_type="10-K",
            source_date=date(2026, 1, 1),
            section="Risk Factors",
            text="unrelated chunk",
            embedding=_vector(dim, {500: 1.0}),
        )
        await postgres_db.save_chunks([far, close])

        results = await postgres_db.search_chunks("AAPL", query, limit=10)

        assert [c.chunk_id for c in results] == ["close-1", "far-1"]

    async def test_search_is_scoped_to_ticker(self, postgres_db):
        dim = 1024
        query = _vector(dim, {0: 1.0})
        aapl_chunk = RetrievalChunk(
            chunk_id="aapl-1",
            ticker="AAPL",
            source_type="10-K",
            source_date=date(2026, 1, 1),
            section=None,
            text="aapl chunk",
            embedding=_vector(dim, {0: 1.0}),
        )
        msft_chunk = RetrievalChunk(
            chunk_id="msft-1",
            ticker="MSFT",
            source_type="10-K",
            source_date=date(2026, 1, 1),
            section=None,
            text="msft chunk",
            embedding=_vector(dim, {0: 1.0}),
        )
        await postgres_db.save_chunks([aapl_chunk, msft_chunk])

        results = await postgres_db.search_chunks("AAPL", query, limit=10)

        assert [c.chunk_id for c in results] == ["aapl-1"]

    async def test_existing_chunk_ids_reports_only_what_is_stored(self, postgres_db):
        """Drives the ingest's skip-what-we-have check, so the `= ANY($1)`
        lookup is exercised against real Postgres rather than only the fake."""
        stored = RetrievalChunk(
            chunk_id="AAPL:10-K:2026-01-01:0",
            ticker="AAPL",
            source_type="10-K",
            source_date=date(2026, 1, 1),
            section=None,
            text="already ingested",
            embedding=_vector(1024, {0: 1.0}),
        )
        await postgres_db.save_chunks([stored])

        found = await postgres_db.existing_chunk_ids(
            [stored.chunk_id, "AAPL:10-K:2026-01-01:1"]
        )

        assert found == {stored.chunk_id}

    async def test_existing_chunk_ids_handles_an_empty_request(self, postgres_db):
        assert await postgres_db.existing_chunk_ids([]) == set()


class TestForecasts:
    def _report(self) -> ForecastReport:
        claim = AgentClaim(
            agent="technicals",
            ticker="AAPL",
            direction="bullish",
            magnitude_bps=150,
            confidence=0.7,
            regime_label="Trending Bull",
            regime_changed=False,
            evidence=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
            falsifiers=["reversal"],
        )
        return ForecastReport(
            run_id=uuid4(),
            ticker="AAPL",
            horizon_days=21,
            generated_at=datetime(2026, 1, 1, 12, 0, 0),
            distribution=Distribution(bullish_pct=60, neutral_pct=25, bearish_pct=15),
            recommendation="bullish_lean",
            overall_confidence=0.7,
            escalate_to_analyst=False,
            thesis="Sample thesis referencing the computed distribution.",
            agent_claims=[claim],
            critique_log=["[pre-critique] technicals: bullish"],
            citations=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
        )

    async def test_round_trips_full_report_including_dissent_log(self, postgres_db):
        report = self._report()

        await postgres_db.save_forecast(report)
        fetched = await postgres_db.get_forecast(str(report.run_id))

        assert fetched == report
        assert fetched.critique_log == report.critique_log
        assert fetched.agent_claims == report.agent_claims

    async def test_unknown_run_id_returns_none(self, postgres_db):
        assert await postgres_db.get_forecast(str(uuid4())) is None

    async def test_pending_evaluation_lifecycle(self, postgres_db):
        report = self._report()
        await postgres_db.save_forecast(report)

        not_yet_due = await postgres_db.list_pending_evaluations(
            as_of=report.generated_at.date() + timedelta(days=1)
        )
        assert report.run_id not in {r.run_id for r in not_yet_due}

        due = await postgres_db.list_pending_evaluations(
            as_of=report.generated_at.date() + timedelta(days=report.horizon_days + 1)
        )
        assert report.run_id in {r.run_id for r in due}

        await postgres_db.record_realized_outcome(
            str(report.run_id), realized_return_bps=250.0, brier_score=0.18
        )

        still_due = await postgres_db.list_pending_evaluations(
            as_of=report.generated_at.date() + timedelta(days=report.horizon_days + 1)
        )
        assert report.run_id not in {r.run_id for r in still_due}
