"""Shared audit-log helpers for specialist and meta-agent traces.

Every agent logs through `log_event` so a forecast run is greppable as
`<agent> called` / `<agent> thinking` / `<agent> decision` lines with the
same key=value fields. Messages stay on one line so `make logs` remains
readable.
"""

from __future__ import annotations

import logging
from typing import Any

from equity_ensemble.schemas.models import AgentClaim

_SNIPPET_LIMIT = 80


def format_fields(**fields: Any) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        parts.append(f"{key}={_render(value)}")
    return " ".join(parts)


def log_event(
    logger: logging.Logger,
    agent: str,
    event: str,
    *,
    ticker: str,
    **fields: Any,
) -> None:
    """Emit one audit line: `<agent> <event> ticker=... key=value ...`."""
    suffix = format_fields(**fields)
    message = f"{agent} {event} ticker={ticker}"
    if suffix:
        message = f"{message} {suffix}"
    logger.info(message)


def summarize_claim(claim: AgentClaim | None) -> str:
    if claim is None:
        return "unavailable"
    parts = [
        f"{claim.agent}:{claim.direction}",
        f"{claim.magnitude_bps}bps",
        f"conf={claim.confidence:.2f}",
    ]
    if claim.regime_label:
        changed = "changed" if claim.regime_changed else "unchanged"
        parts.append(f"regime={claim.regime_label}({changed})")
    if claim.sentiment_score is not None:
        parts.append(f"sentiment={claim.sentiment_score}")
    if claim.key_risk_flags:
        parts.append(f"risks={','.join(claim.key_risk_flags)}")
    return "/".join(parts)


def summarize_evidence(claim: AgentClaim) -> str:
    if not claim.evidence:
        return "[]"
    rendered = []
    for item in claim.evidence:
        snippet = _collapse(item.snippet)
        if len(snippet) > _SNIPPET_LIMIT:
            snippet = snippet[: _SNIPPET_LIMIT - 3] + "..."
        rendered.append(f"{item.source}({item.date.isoformat()}):{snippet}")
    return " | ".join(rendered)


def _render(value: Any) -> str:
    if value is None:
        rendered = "none"
    elif isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, float):
        rendered = f"{value:.4g}"
    elif isinstance(value, dict):
        rendered = ",".join(f"{key}:{_render(item)}" for key, item in value.items()) or "{}"
    elif isinstance(value, (list, tuple)):
        rendered = ",".join(_render(item) for item in value) if value else "[]"
    else:
        rendered = str(value)
    return _collapse(rendered)


def _collapse(value: str) -> str:
    return " ".join(value.split())
