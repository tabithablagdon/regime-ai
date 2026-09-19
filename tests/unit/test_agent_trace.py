from __future__ import annotations

import logging

from equity_ensemble.agents.trace import (
    format_fields,
    log_event,
    summarize_claim,
    summarize_evidence,
)


def test_format_fields_collapses_whitespace_and_renders_none():
    rendered = format_fields(reason="line one\nline two", missing=None, flag=False)
    assert rendered == "reason=line one line two missing=none flag=false"


def test_summarize_claim_includes_direction_and_regime(technicals_claim_factory):
    claim = technicals_claim_factory(direction="bullish", confidence=0.8, regime_changed=True)
    summary = summarize_claim(claim)
    assert "technicals:bullish" in summary
    assert "conf=0.80" in summary
    assert "regime=Trending Bull(changed)" in summary


def test_summarize_claim_unavailable():
    assert summarize_claim(None) == "unavailable"


def test_summarize_evidence_truncates_long_snippets(technicals_claim_factory):
    from datetime import date

    from equity_ensemble.schemas.models import Evidence

    claim = technicals_claim_factory(
        evidence=[
            Evidence(source="10-Q", date=date(2026, 8, 1), snippet="x" * 200),
        ]
    )
    rendered = summarize_evidence(claim)
    assert rendered.startswith("10-Q(2026-08-01):")
    assert rendered.endswith("...")
    assert len(rendered) < 200


def test_log_event_writes_greppable_audit_line(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("equity_ensemble.agents.trace")

    log_event(
        logger,
        "technicals",
        "decision",
        ticker="AAPL",
        direction="bullish",
        magnitude_bps=150,
    )

    assert "technicals decision ticker=AAPL direction=bullish magnitude_bps=150" in caplog.text
