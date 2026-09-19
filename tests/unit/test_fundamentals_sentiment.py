from __future__ import annotations

from equity_ensemble.agents.fundamentals_sentiment import _select_relevant_filings


def _filing(filing_type: str, date_str: str) -> dict:
    return {"type": filing_type, "date": date_str, "section": None, "text": "x"}


class TestSelectRelevantFilings:
    def test_empty_input_returns_empty(self):
        assert _select_relevant_filings([]) == []

    def test_keeps_most_recent_10k_and_everything_since(self):
        filings = [
            _filing("10-K", "2024-10-01"),  # older 10-K, should be dropped
            _filing("10-Q", "2025-01-15"),  # before the newer 10-K, dropped
            _filing("10-K", "2025-10-01"),  # most recent 10-K, the cutoff
            _filing("8-K", "2025-11-01"),
            _filing("10-Q", "2026-01-15"),
        ]

        result = _select_relevant_filings(filings)

        dates = {f["date"] for f in result}
        assert dates == {"2025-10-01", "2025-11-01", "2026-01-15"}

    def test_no_10k_falls_back_to_everything(self):
        filings = [_filing("8-K", "2026-01-01"), _filing("10-Q", "2026-02-01")]

        result = _select_relevant_filings(filings)

        assert len(result) == 2

    def test_ordered_most_recent_first(self):
        filings = [
            _filing("10-K", "2025-10-01"),
            _filing("8-K", "2025-11-01"),
            _filing("10-Q", "2026-01-15"),
        ]

        result = _select_relevant_filings(filings)

        assert [f["date"] for f in result] == ["2026-01-15", "2025-11-01", "2025-10-01"]

    def test_single_10k_only(self):
        filings = [_filing("10-K", "2025-10-01")]

        result = _select_relevant_filings(filings)

        assert len(result) == 1
