from __future__ import annotations

import pytest

from equity_ensemble.eval.brier_score import (
    _find_bar_on_or_after,
    brier_score,
    realized_bucket_from_return_bps,
)
from equity_ensemble.schemas.models import Distribution


class TestBrierScore:
    def test_perfect_confident_correct_forecast_scores_zero(self):
        dist = Distribution(bullish_pct=100, neutral_pct=0, bearish_pct=0)
        assert brier_score(dist, "bullish") == pytest.approx(0.0)

    def test_perfect_confident_wrong_forecast_scores_one(self):
        dist = Distribution(bullish_pct=100, neutral_pct=0, bearish_pct=0)
        assert brier_score(dist, "bearish") == pytest.approx(1.0)

    def test_uniform_forecast_beats_neither_way_scores_around_one_third(self):
        dist = Distribution(bullish_pct=100 / 3, neutral_pct=100 / 3, bearish_pct=100 / 3)
        assert brier_score(dist, "bullish") == pytest.approx(1 / 3, abs=1e-6)

    def test_invalid_bucket_raises(self):
        dist = Distribution(bullish_pct=50, neutral_pct=30, bearish_pct=20)
        with pytest.raises(ValueError):
            brier_score(dist, "sideways")


class TestRealizedBucket:
    def test_strongly_positive_return_is_bullish(self):
        assert realized_bucket_from_return_bps(200) == "bullish"

    def test_strongly_negative_return_is_bearish(self):
        assert realized_bucket_from_return_bps(-200) == "bearish"

    def test_small_return_is_neutral(self):
        assert realized_bucket_from_return_bps(10) == "neutral"


class TestFindBarOnOrAfter:
    def test_finds_closest_bar_on_or_after_target(self):
        bars = [
            {"date": "2026-01-01", "close": 100},
            {"date": "2026-01-05", "close": 105},
            {"date": "2026-01-10", "close": 110},
        ]
        from datetime import date

        bar = _find_bar_on_or_after(bars, date(2026, 1, 3))
        assert bar["date"] == "2026-01-05"

    def test_returns_none_if_no_bar_covers_target(self):
        from datetime import date

        bars = [{"date": "2026-01-01", "close": 100}]
        assert _find_bar_on_or_after(bars, date(2026, 6, 1)) is None
