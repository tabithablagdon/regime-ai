"""Single wrapper around every Financial Modeling Prep (FMP) endpoint the
system uses (PRD §5, §13).

`FMPClient` is the protocol both specialist agents code against.
`RecordedFMPClient` replays fixture JSON from `tests/fixtures/fmp/` so Track A
(Technicals) and Track B (Fundamentals/Sentiment) can be built and tested
without live API access or quota. `RealFMPClient` is the live implementation.

`RealFMPClient` targets FMP's "stable" API (`/stable/...`). FMP retired the
legacy `/api/v3/...` family on 2025-08-31 for all but grandfathered
subscriptions — every `/api/v3` call now 403s with "Legacy Endpoint" for a
new key, which is how this was caught (see the FMP developer docs at
https://site.financialmodelingprep.com/developer/docs for the current set).
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Modern SEC filings are inline-XBRL: a large `display:none` block up front
# holds every tagged fact (fiscal year, CIK, dei:*, us-gaap:* references) as
# raw, non-prose text. Naive tag-stripping alone leaves this in — it isn't
# hidden from a regex, only from a renderer — and it can be tens of KB of
# pure noise sitting before any real content. Strip it before stripping tags.
_HIDDEN_DIV_RE = re.compile(
    r'<div[^>]*style="[^"]*display\s*:\s*none[^"]*"[^>]*>.*?</div>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_html_tags(html_text: str) -> str:
    """Naive HTML -> text extraction for filing bodies. Good enough to feed
    the chunker; swap for a real HTML parser (e.g. BeautifulSoup) if section
    boundaries need to be recovered more precisely than 'whole document'."""
    cleaned = _HIDDEN_DIV_RE.sub(" ", html_text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _date_range(days_back: int) -> tuple[str, str]:
    """`(from, to)` ISO date strings for FMP's stable-API date-range params."""
    today = datetime.now(UTC).date()
    return (today - timedelta(days=days_back)).isoformat(), today.isoformat()


def _parse_fmp_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=UTC)


class TickerNotFoundError(RuntimeError):
    """Raised when a ticker fails FMP profile lookup (PRD §4.2 step 2)."""


class FMPClient(Protocol):
    async def get_profile(self, ticker: str) -> dict[str, Any]:
        """Company profile lookup. Used to validate the ticker exists and is
        a supported U.S. equity before the graph fans out (PRD §4.2)."""
        ...

    async def get_ohlcv(self, ticker: str, days: int = 504) -> list[dict[str, Any]]:
        """~2 years of daily OHLCV (Technicals agent, PRD §6.1)."""
        ...

    async def get_technical_indicators(
        self, ticker: str, indicator: str, period: int = 14
    ) -> list[dict[str, Any]]:
        """One of sma/ema/rsi/macd/bbands/adx (Technicals agent, PRD §6.1)."""
        ...

    async def get_options_iv(self, ticker: str) -> dict[str, Any] | None:
        """Options implied volatility, where available (Technicals agent)."""
        ...

    async def get_filings(
        self, ticker: str, filing_types: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    ) -> list[dict[str, Any]]:
        """Filing text for the given types (Fundamentals/Sentiment agent,
        PRD §6.2). Falls back to SEC EDGAR full-text search when a filing
        isn't mirrored by FMP — implemented by the real client. Each row:
        `{"type": str, "date": str, "section": str | None, "text": str}`."""
        ...

    async def get_news(self, ticker: str, days: int = 14) -> list[dict[str, Any]]:
        """Last N days of company news (Fundamentals/Sentiment agent)."""
        ...

    async def get_estimate_revisions(self, ticker: str) -> list[dict[str, Any]]:
        """Consensus estimate revisions, where FMP exposes them."""
        ...


class RealFMPClient:
    """Live implementation. Endpoint bodies are filled in by Track A
    (market-data/indicator methods) and Track B (filings/news methods) —
    each track only ever adds methods here, never changes the protocol
    shape above without a reviewed schema change."""

    def __init__(
        self, *, api_key: str | None = None, client: httpx.AsyncClient | None = None
    ) -> None:
        self._api_key = api_key or os.environ["FMP_API_KEY"]
        self._client = client or httpx.AsyncClient(base_url=FMP_BASE_URL, timeout=30.0)

    async def _get(self, path: str, **params: Any) -> Any:
        resp = await self._client.get(path, params={**params, "apikey": self._api_key})
        resp.raise_for_status()
        return resp.json()

    async def get_profile(self, ticker: str) -> dict[str, Any]:
        data = await self._get("/profile", symbol=ticker)
        if not data:
            raise TickerNotFoundError(ticker)
        return data[0]

    async def get_ohlcv(self, ticker: str, days: int = 504) -> list[dict[str, Any]]:
        # Stable API takes a calendar date range, not a trading-day count —
        # over-fetch a bit (days are trading days, ~252/year) so at least
        # `days` bars come back after weekends/holidays are excluded.
        date_from, date_to = _date_range(int(days * 1.5))
        return await self._get(
            "/historical-price-eod/full", symbol=ticker, **{"from": date_from, "to": date_to}
        )

    async def get_technical_indicators(
        self, ticker: str, indicator: str, period: int = 14
    ) -> list[dict[str, Any]]:
        return await self._get(
            f"/technical-indicators/{indicator}",
            symbol=ticker,
            periodLength=period,
            timeframe="1day",
        )

    async def get_options_iv(self, ticker: str) -> dict[str, Any] | None:
        raise NotImplementedError(
            "Track A: implement once the FMP options-IV endpoint is confirmed"
        )

    async def get_filings(
        self, ticker: str, filing_types: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    ) -> list[dict[str, Any]]:
        date_from, date_to = _date_range(730)  # 2 years — comfortably covers 10-Ks + 10-Qs
        rows = await self._get(
            "/sec-filings-search/symbol",
            symbol=ticker,
            **{"from": date_from, "to": date_to},
        )
        filings = []
        for row in rows:
            if row.get("formType") not in filing_types:
                continue
            filings.append(await self._parse_filing_row(row))
        if not filings:
            raise NotImplementedError(
                f"FMP has no mirrored {filing_types} filings for {ticker} — implement the "
                "documented SEC EDGAR full-text-search fallback (PRD §5) here; confirm "
                "coverage during M2 per PRD §16"
            )
        return filings

    async def _parse_filing_row(self, row: dict[str, Any]) -> dict[str, Any]:
        link = row.get("finalLink") or row.get("link")
        text = await self._fetch_filing_text(link) if link else ""
        return {
            "type": row.get("formType"),
            "date": row.get("filingDate") or row.get("acceptedDate"),
            "section": None,
            "text": text,
        }

    async def _fetch_filing_text(self, url: str) -> str:
        # SEC EDGAR's fair-access policy 403s any request without a
        # descriptive User-Agent (name + contact) — httpx's default
        # ("python-httpx/...") gets blocked. See
        # https://www.sec.gov/os/webmaster-faq#developers
        resp = await self._client.get(
            url,
            headers={"User-Agent": "equity-ensemble tabitha@helmhealth.com"},
        )
        resp.raise_for_status()
        return _strip_html_tags(resp.text)

    async def get_news(self, ticker: str, days: int = 14) -> list[dict[str, Any]]:
        rows = await self._get("/news/stock", symbols=ticker, limit=100)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [row for row in rows if _parse_fmp_datetime(row.get("publishedDate")) >= cutoff]

    async def get_estimate_revisions(self, ticker: str) -> list[dict[str, Any]]:
        return await self._get("/analyst-estimates", symbol=ticker, period="quarter")


class RecordedFMPClient:
    """Fixture-replaying fake. Reads `<fixtures_dir>/<method>/<ticker>.json`.

    Record a real response once (`json.dump`'d FMP payload) and every test
    for that track runs offline and deterministically against it.
    """

    def __init__(self, fixtures_dir: str | Path) -> None:
        self._dir = Path(fixtures_dir)

    def _load(self, method: str, ticker: str) -> Any:
        path = self._dir / method / f"{ticker.upper()}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no recorded fixture at {path} — record one with the real FMPClient first"
            )
        return json.loads(path.read_text())

    async def get_profile(self, ticker: str) -> dict[str, Any]:
        return self._load("get_profile", ticker)

    async def get_ohlcv(self, ticker: str, days: int = 504) -> list[dict[str, Any]]:
        return self._load("get_ohlcv", ticker)

    async def get_technical_indicators(
        self, ticker: str, indicator: str, period: int = 14
    ) -> list[dict[str, Any]]:
        return self._load(f"get_technical_indicators_{indicator}", ticker)

    async def get_options_iv(self, ticker: str) -> dict[str, Any] | None:
        try:
            return self._load("get_options_iv", ticker)
        except FileNotFoundError:
            return None

    async def get_filings(
        self, ticker: str, filing_types: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    ) -> list[dict[str, Any]]:
        return self._load("get_filings", ticker)

    async def get_news(self, ticker: str, days: int = 14) -> list[dict[str, Any]]:
        return self._load("get_news", ticker)

    async def get_estimate_revisions(self, ticker: str) -> list[dict[str, Any]]:
        try:
            return self._load("get_estimate_revisions", ticker)
        except FileNotFoundError:
            return []
