"""Forecast calibration eval job — Track E (PRD §12).

Scheduled script: for every forecast whose horizon has elapsed and that
doesn't yet have a realized outcome, pull the realized closing price,
compute the return vs. the price at generation time, bucket it, score it
against the stored distribution, and write both back. Target: beat the
~0.33 random-baseline Brier score.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from equity_ensemble.agents.meta_agent import NEUTRAL_BAND_BPS
from equity_ensemble.data.fmp_client import FMPClient
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import Distribution, ForecastReport


def brier_score(distribution: Distribution, realized_bucket: str) -> float:
    """Multi-category Brier score, PRD §12: `0.5 * sum((p_i - o_i)^2)` over
    the three buckets, where `o_i` is 1 for the realized bucket and 0
    otherwise. A uniform-random 1/3-1/3-1/3 forecast scores ~0.33 — the
    baseline the PRD asks the ensemble to beat."""
    probabilities = {
        "bullish": distribution.bullish_pct / 100.0,
        "neutral": distribution.neutral_pct / 100.0,
        "bearish": distribution.bearish_pct / 100.0,
    }
    if realized_bucket not in probabilities:
        raise ValueError(
            f"realized_bucket must be one of {sorted(probabilities)}, got {realized_bucket!r}"
        )
    return 0.5 * sum(
        (p - (1.0 if bucket == realized_bucket else 0.0)) ** 2
        for bucket, p in probabilities.items()
    )


def realized_bucket_from_return_bps(realized_return_bps: float) -> str:
    """Buckets a realized return the same way MetaAgent.fit_distribution
    buckets its forecast, so scoring compares like with like."""
    if realized_return_bps > NEUTRAL_BAND_BPS:
        return "bullish"
    if realized_return_bps < -NEUTRAL_BAND_BPS:
        return "bearish"
    return "neutral"


def _parse_bar_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _find_bar_on_or_after(bars: list[dict], target: date) -> dict | None:
    candidates = [b for b in bars if _parse_bar_date(b["date"]) >= target]
    return min(candidates, key=lambda b: _parse_bar_date(b["date"])) if candidates else None


async def compute_realized_return_bps(fmp: FMPClient, report: ForecastReport) -> float:
    """Realized return over the forecast horizon, in basis points. Horizon
    is treated as calendar days for MVP (documented simplification — see
    PRD §12); a trading-day calendar can replace this once real latencies
    make the distinction matter."""
    bars = sorted(await fmp.get_ohlcv(report.ticker), key=lambda b: b["date"])
    start_date = report.generated_at.date()
    end_date = start_date + timedelta(days=report.horizon_days)

    start_bar = _find_bar_on_or_after(bars, start_date)
    end_bar = _find_bar_on_or_after(bars, end_date)
    if start_bar is None or end_bar is None:
        raise ValueError(
            f"insufficient OHLCV history to score forecast {report.run_id} for {report.ticker}"
        )

    start_price, end_price = float(start_bar["close"]), float(end_bar["close"])
    return (end_price - start_price) / start_price * 10_000


async def run_scoring_job(db: Database, fmp: FMPClient, *, as_of: date | None = None) -> list[str]:
    """Scores every pending forecast and writes the outcome back. Returns
    the run_ids that were scored."""
    scored: list[str] = []
    for report in await db.list_pending_evaluations(as_of=as_of):
        realized_return_bps = await compute_realized_return_bps(fmp, report)
        bucket = realized_bucket_from_return_bps(realized_return_bps)
        score = brier_score(report.distribution, bucket)
        await db.record_realized_outcome(
            str(report.run_id), realized_return_bps=realized_return_bps, brier_score=score
        )
        scored.append(str(report.run_id))
    return scored
