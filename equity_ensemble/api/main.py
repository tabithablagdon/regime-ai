"""FastAPI app — owned by Track D (PRD §10.1).

POST /forecast  { "ticker": "AAPL", "horizon_days": 21 }
  -> 200 { ...ForecastReport fields..., "report_markdown": "..." }

Ticker validation (existence via FMP profile lookup) happens before the
graph fans out (PRD §4.2 step 2) — reject unknown tickers immediately.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Equity Forecast Ensemble", version="0.1.0")


class ForecastRequest(BaseModel):
    ticker: str
    horizon_days: int = 21


@app.post("/forecast")
async def forecast(request: ForecastRequest) -> dict:
    raise NotImplementedError(
        "Track D: validate ticker via FMPClient.get_profile, run the compiled graph, "
        "return ForecastReport + report_markdown"
    )
