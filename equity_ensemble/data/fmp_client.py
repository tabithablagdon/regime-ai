"""Single wrapper around every Financial Modeling Prep (FMP) endpoint the
system uses (PRD §5, §13).

`FMPClient` is the protocol both specialist agents code against.
`RecordedFMPClient` replays fixture JSON from `tests/fixtures/fmp/` so Track A
(Technicals) and Track B (Fundamentals/Sentiment) can be built and tested
without live API access or quota. `RealFMPClient` is the live implementation,
filled in as each track needs a specific endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


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
        """Filing links for the given types (Fundamentals/Sentiment agent,
        PRD §6.2). Falls back to SEC EDGAR full-text search when a filing
        isn't mirrored by FMP — implemented by the real client."""
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
        data = await self._get(f"/profile/{ticker}")
        if not data:
            raise TickerNotFoundError(ticker)
        return data[0]

    async def get_ohlcv(self, ticker: str, days: int = 504) -> list[dict[str, Any]]:
        data = await self._get(f"/historical-price-full/{ticker}", timeseries=days)
        return data.get("historical", [])

    async def get_technical_indicators(
        self, ticker: str, indicator: str, period: int = 14
    ) -> list[dict[str, Any]]:
        return await self._get(
            f"/technical_indicator/daily/{ticker}", type=indicator, period=period
        )

    async def get_options_iv(self, ticker: str) -> dict[str, Any] | None:
        raise NotImplementedError(
            "Track A: implement once the FMP options-IV endpoint is confirmed"
        )

    async def get_filings(
        self, ticker: str, filing_types: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Track B: implement filing-links endpoint + SEC EDGAR fallback")

    async def get_news(self, ticker: str, days: int = 14) -> list[dict[str, Any]]:
        raise NotImplementedError("Track B: implement stock-news endpoint")

    async def get_estimate_revisions(self, ticker: str) -> list[dict[str, Any]]:
        raise NotImplementedError("Track B: implement consensus estimate revisions endpoint")


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
