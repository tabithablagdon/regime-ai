from __future__ import annotations

import pytest

from equity_ensemble.agents.technicals import (
    classify_regime,
    price_vs_moving_average_pct,
    realized_volatility_from_closes,
)


class TestClassifyRegime:
    def test_trending_up_with_high_adx_is_trending_bull(self):
        label = classify_regime(realized_volatility=0.2, adx=30, price_vs_moving_average_pct=5.0)
        assert label == "Trending Bull"

    def test_trending_down_with_high_adx_is_trending_bear(self):
        label = classify_regime(realized_volatility=0.2, adx=30, price_vs_moving_average_pct=-5.0)
        assert label == "Trending Bear"

    def test_low_adx_high_volatility_is_choppy(self):
        label = classify_regime(realized_volatility=0.5, adx=10, price_vs_moving_average_pct=1.0)
        assert label == "Volatile/Choppy"

    def test_low_adx_low_volatility_is_mean_reverting(self):
        label = classify_regime(realized_volatility=0.1, adx=10, price_vs_moving_average_pct=1.0)
        assert label == "Mean-Reverting"

    def test_adx_exactly_at_threshold_counts_as_trending(self):
        label = classify_regime(realized_volatility=0.1, adx=25.0, price_vs_moving_average_pct=1.0)
        assert label == "Trending Bull"


class TestRealizedVolatility:
    def test_constant_prices_have_zero_volatility(self):
        closes = [100.0] * 30
        assert realized_volatility_from_closes(closes) == pytest.approx(0.0, abs=1e-9)

    def test_more_volatile_series_has_higher_score(self):
        calm = [100.0 + (i % 2) * 0.1 for i in range(30)]
        wild = [100.0 + (i % 2) * 10 for i in range(30)]
        assert realized_volatility_from_closes(wild) > realized_volatility_from_closes(calm)

    def test_requires_at_least_two_closes(self):
        with pytest.raises(ValueError):
            realized_volatility_from_closes([100.0])


class TestPriceVsMovingAverage:
    def test_price_above_average_is_positive(self):
        assert price_vs_moving_average_pct(110.0, 100.0) == pytest.approx(10.0)

    def test_price_below_average_is_negative(self):
        assert price_vs_moving_average_pct(90.0, 100.0) == pytest.approx(-10.0)

    def test_price_at_average_is_zero(self):
        assert price_vs_moving_average_pct(100.0, 100.0) == pytest.approx(0.0)
