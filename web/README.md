# Regime AI — Web UI

A small Next.js app: enter a ticker, get the equity forecast report
rendered in the browser. Talks to the FastAPI backend
(`equity_ensemble.api.main`) via a server-side proxy route
(`app/api/forecast/route.ts`) — the browser never calls the backend
directly.

## Prerequisites

The backend must be running first. From the repo root:

```bash
cd ..
make up
```

## Run

```bash
npm install
cp .env.local.example .env.local   # defaults to http://127.0.0.1:8000, edit if needed
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), enter a ticker (e.g.
`AAPL`), and submit.

## What you'll see

- **A rendered report** — probability distribution, thesis, both agents'
  raw claims side by side (the dissent log), citations — if the backend
  successfully produces a `ForecastReport`.
- **"Ticker not found"** (inline, near the input) — if the ticker doesn't
  exist or isn't a supported U.S. equity.
- **"Analysis engine unavailable"** (a banner) — if the backend's
  reasoning pipeline fails for any reason (most commonly: `OPENROUTER_API_KEY`,
  `FMP_API_KEY`, or `VOYAGE_API_KEY` in the backend's `.env` aren't real
  keys yet), or if the backend isn't reachable at all. This is the expected
  state until real vendor keys are configured — see the repo root's
  `README.md` for how to get them.

## Structure

```
app/
  page.tsx                 # the page: form + request state + result rendering
  api/forecast/route.ts    # server-side proxy to the FastAPI backend
  globals.css               # design tokens (dataviz skill's diverging palette
                             #   for bullish/neutral/bearish + chart chrome)
components/
  TickerForm.tsx
  DistributionMeter.tsx     # the bullish/neutral/bearish probability meter
  AgentClaimCard.tsx        # one specialist's raw claim
  ForecastReportView.tsx    # composes the full report
  EngineUnavailableBanner.tsx
lib/
  types.ts                  # TypeScript types mirroring
                             #   equity_ensemble/schemas/models.py
```

## Build

```bash
npm run build
```
