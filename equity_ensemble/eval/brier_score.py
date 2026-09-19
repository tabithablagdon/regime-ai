"""Forecast calibration eval job — owned by Track E (PRD §12).

Scheduled script: for every forecast whose horizon has elapsed and that
doesn't yet have a realized_return, pull the realized return, compute the
Brier score against the stored distribution, write both back to the
`forecasts` table. Target: beat the ~0.33 random-baseline Brier score.
"""

from __future__ import annotations

from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import Distribution


def brier_score(distribution: Distribution, realized_bucket: str) -> float:
    """`realized_bucket` is one of "bullish"/"neutral"/"bearish" — whichever
    the realized return actually fell into. Pure function, unit-testable
    with synthetic forecast/outcome pairs."""
    raise NotImplementedError("Track E: implement the Brier score calculation")


async def run_scoring_job(db: Database) -> None:
    raise NotImplementedError(
        "Track E: find forecasts past horizon with no realized_return, pull realized "
        "return, compute brier_score, write both back via db"
    )
