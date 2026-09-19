from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from equity_ensemble.data.fmp_client import RecordedFMPClient
from equity_ensemble.persistence.db import FakeDatabase
from equity_ensemble.schemas.models import AgentClaim, Evidence

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_db() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def recorded_fmp() -> RecordedFMPClient:
    return RecordedFMPClient(FIXTURES_DIR / "fmp")


def make_evidence(
    source: str = "Q2 10-Q", as_of: date = date(2026, 8, 1), snippet: str = "sample"
) -> Evidence:
    return Evidence(source=source, date=as_of, snippet=snippet)


@pytest.fixture
def technicals_claim_factory():
    def _make(**overrides) -> AgentClaim:
        defaults = dict(
            agent="technicals",
            ticker="AAPL",
            direction="bullish",
            magnitude_bps=150,
            confidence=0.7,
            regime_label="Trending Bull",
            regime_changed=False,
            evidence=[make_evidence()],
            falsifiers=["ADX drops below 20 within 5 sessions"],
        )
        defaults.update(overrides)
        return AgentClaim(**defaults)

    return _make


@pytest.fixture
def fundamentals_claim_factory():
    def _make(**overrides) -> AgentClaim:
        defaults = dict(
            agent="fundamentals_sentiment",
            ticker="AAPL",
            direction="bearish",
            magnitude_bps=120,
            confidence=0.65,
            sentiment_score=40,
            key_risk_flags=["guidance_cut"],
            evidence=[make_evidence(source="Q2 earnings call transcript")],
            falsifiers=["Next 8-K reaffirms prior guidance"],
        )
        defaults.update(overrides)
        return AgentClaim(**defaults)

    return _make


@pytest.fixture
def new_run_id() -> str:
    return str(uuid4())


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 19, 12, 0, 0)
