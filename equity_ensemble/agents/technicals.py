"""Technicals Agent — Track A (PRD §6.1).

ReAct loop: read the ticker's last-known regime (long-term memory) -> fetch
today's price/indicator values -> classify the regime in code -> have the
LLM narrate that computed result into a claim (the LLM never derives the
regime itself — its output is overridden with the computed values before
being returned, so a mismatched narration can't silently corrupt the claim).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from equity_ensemble.agents.base import BaseSpecialistAgent, run_with_validation_retry
from equity_ensemble.data.fmp_client import FMPClient
from equity_ensemble.llm.client import LLMClient
from equity_ensemble.persistence.db import Database
from equity_ensemble.schemas.models import AgentClaim

REGIME_LABELS = ("Trending Bull", "Trending Bear", "Mean-Reverting", "Volatile/Choppy")

# Classification thresholds (PRD §6.1: computed in code, not by the LLM).
ADX_TRENDING_THRESHOLD = 25.0
VOLATILITY_CHOPPY_THRESHOLD = 0.35  # annualized realized volatility
TRADING_DAYS_PER_YEAR = 252
ADX_PERIOD = 14
SMA_PERIOD = 50
MIN_CLOSES_REQUIRED = 30

TECHNICALS_SYSTEM_PROMPT = (
    "You are the Technicals specialist in an equity forecast ensemble. A "
    "rule-based classifier (not you) has already determined the price/volume "
    "regime below from real indicator data — your job is only to interpret and "
    "narrate that computed result into a claim, never to recompute or "
    "contradict the regime label you are given. State a directional view and "
    "magnitude consistent with the regime, and list concrete falsifiers."
)


def classify_regime(
    *,
    realized_volatility: float,
    adx: float,
    price_vs_moving_average_pct: float,
) -> str:
    """Pure, deterministic regime classifier (PRD §6.1). No LLM call.

    ADX >= threshold means the market is trending; direction (bull/bear) is
    then read off price-vs-moving-average. Below the ADX threshold, realized
    volatility distinguishes a quiet mean-reverting tape from a choppy one.
    """
    if adx >= ADX_TRENDING_THRESHOLD:
        return "Trending Bull" if price_vs_moving_average_pct >= 0 else "Trending Bear"
    if realized_volatility >= VOLATILITY_CHOPPY_THRESHOLD:
        return "Volatile/Choppy"
    return "Mean-Reverting"


def realized_volatility_from_closes(
    closes: list[float], *, annualization_factor: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized realized volatility from a series of daily closes, oldest
    first."""
    if len(closes) < 2:
        raise ValueError("need at least 2 closes to compute realized volatility")
    log_returns = np.diff(np.log(np.asarray(closes, dtype=float)))
    return float(np.std(log_returns, ddof=1) * np.sqrt(annualization_factor))


def price_vs_moving_average_pct(price: float, moving_average: float) -> float:
    return (price - moving_average) / moving_average * 100.0


def _latest_indicator_value(series: list[dict], key: str) -> float:
    if not series:
        raise ValueError(f"empty indicator series for '{key}'")
    latest = max(series, key=lambda row: row["date"])
    return float(latest[key])


class TechnicalsAgent(BaseSpecialistAgent[AgentClaim]):
    def __init__(self, llm: LLMClient, fmp: FMPClient, db: Database) -> None:
        super().__init__(llm)
        self.fmp = fmp
        self.db = db

    async def run(self, ticker: str, horizon_days: int) -> AgentClaim:
        ticker = ticker.upper()
        previous_regime = await self.db.get_last_regime(ticker)

        ohlcv = await self.fmp.get_ohlcv(ticker)
        if len(ohlcv) < MIN_CLOSES_REQUIRED:
            raise ValueError(
                f"insufficient OHLCV history for {ticker}: "
                f"got {len(ohlcv)} bars, need at least {MIN_CLOSES_REQUIRED}"
            )
        bars = sorted(ohlcv, key=lambda bar: bar["date"])
        closes = [float(bar["close"]) for bar in bars]
        latest_close = closes[-1]

        adx_series = await self.fmp.get_technical_indicators(ticker, "adx", period=ADX_PERIOD)
        sma_series = await self.fmp.get_technical_indicators(ticker, "sma", period=SMA_PERIOD)
        latest_adx = _latest_indicator_value(adx_series, "adx")
        latest_sma = _latest_indicator_value(sma_series, "sma")

        window = closes[-(TRADING_DAYS_PER_YEAR + 1) :]
        volatility = realized_volatility_from_closes(window)
        pct_vs_sma = price_vs_moving_average_pct(latest_close, latest_sma)

        # Options IV is "where available" (PRD §6.1) — the endpoint itself is
        # still unconfirmed vendor-side (PRD §16), so this is best-effort
        # context for the narration prompt, never something the classifier
        # or the retry loop depends on.
        try:
            options_iv = await self.fmp.get_options_iv(ticker)
        except NotImplementedError:
            options_iv = None

        regime_label = classify_regime(
            realized_volatility=volatility,
            adx=latest_adx,
            price_vs_moving_average_pct=pct_vs_sma,
        )
        regime_changed = previous_regime is not None and previous_regime != regime_label
        await self.db.save_regime(ticker, date.today(), regime_label)

        def build_user_prompt(previous_error: str | None) -> str:
            prompt = (
                f"Ticker: {ticker}\n"
                f"Horizon: {horizon_days} trading days\n"
                f"Computed regime: {regime_label}\n"
                f"Previous regime: {previous_regime or 'none on record'} "
                f"(changed: {regime_changed})\n"
                f"Latest close: {latest_close:.2f}\n"
                f"ADX({ADX_PERIOD}): {latest_adx:.1f}\n"
                f"SMA({SMA_PERIOD}): {latest_sma:.2f} (price is {pct_vs_sma:+.2f}% vs SMA)\n"
                f"Annualized realized volatility: {volatility:.1%}\n"
            )
            if options_iv:
                prompt += f"Options implied volatility: {options_iv}\n"
            if previous_error:
                prompt += (
                    f"\nYour previous response failed schema validation: {previous_error}\n"
                    "Fix it and return a valid claim."
                )
            return prompt

        claim = await run_with_validation_retry(
            llm=self.llm,
            system_prompt=TECHNICALS_SYSTEM_PROMPT,
            build_user_prompt=build_user_prompt,
            response_model=AgentClaim,
        )

        # The regime is computed in code, not by the LLM (PRD §6.1) — enforce
        # that guarantee rather than merely requesting it in the prompt.
        return claim.model_copy(
            update={
                "agent": "technicals",
                "ticker": ticker,
                "regime_label": regime_label,
                "regime_changed": regime_changed,
            }
        )
