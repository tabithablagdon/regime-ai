# regime-ai — Equity Forecast Ensemble

Regime AI is a multi-agent decision-support platform engineered for buy-side
research analysts, quantitative portfolio managers, and risk managers. Given
a single U.S.-listed equity ticker, it produces an auditable forecast report
— a probability distribution over bullish/neutral/bearish outcomes, a
written thesis, and a full dissent log — by running two independent
specialist agents in parallel and reconciling their claims through a
deterministic synthesis pipeline, never a black box.

It is a decision-support tool only. **No code path in this system can
place, size, or route a trade** — that's a hard architectural constraint,
not a UI-level one, and every report says so.

This README covers what the system does, how the pieces fit together, what
each component is responsible for, and how to run it locally. The full
product spec lives in `Equity_Forecast_Ensemble_MVP_PRD.docx`; this
document is the engineering map of what's actually built.

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Components](#components)
  - [Schemas — the shared contract](#schemas--the-shared-contract)
  - [Technicals Agent](#technicals-agent)
  - [Fundamentals/Sentiment Agent](#fundamentalssentiment-agent)
  - [Meta-Agent](#meta-agent)
  - [LLM client](#llm-client)
  - [Data layer](#data-layer)
  - [Persistence](#persistence)
  - [Graph](#graph)
  - [Rendering](#rendering)
  - [API and CLI](#api-and-cli)
  - [Eval](#eval)
- [Guardrails](#guardrails)
- [Running it](#running-it)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [Configuration reference](#configuration-reference)
- [Known limitations](#known-limitations)

## What it does

Send it a ticker and a forecast horizon (default 21 trading days) and it
returns:

- **A probability distribution** — bullish / neutral / bearish percentages,
  fit numerically (NumPy/SciPy), never guessed by an LLM.
- **A written thesis** — generated only after every number is already
  fixed, so the model narrates the computed result instead of inventing one.
- **A dissent log** — both specialist agents' raw, unedited claims, side by
  side, plus the back-and-forth if they disagreed. The blended result never
  silently papers over disagreement.
- **Citations** — every fundamental/sentiment claim traces back to a
  specific filing section + date, or a news article + date.
- **An escalation flag** — `escalate_to_analyst`, forced to `true` whenever
  confidence is low or the two agents still disagree after one round of
  reconsidering each other's evidence. Ambiguous cases get flagged for a
  human, never a confident-sounding guess.

Output is available as a Markdown report (for a person to read) and a JSON
payload (for a downstream system to consume) — via both an HTTP API and a
CLI that share the exact same code path.

## How it works

```
Client (CLI or POST /forecast)
        │
        ▼
Orchestrator (FastAPI / Typer) — validates the ticker via FMP profile lookup
        │
        ▼
LangGraph state graph
        │
        ├──────────────┬───────────────────┐
        ▼              ▼                   │
  Technicals Agent   Fundamentals/          │  (run concurrently)
        │            Sentiment Agent        │
        │              │                    │
        └──────┬───────┘                    │
               ▼                            │
         Meta-Agent (synthesize) ◄──────────┘
   validate → conflict-check → (conditional)
   single critique round → fit distribution
   → generate rationale
               │
               ▼
   ForecastReport → Markdown + JSON → response / stdout + files
```

Two specialist agents each produce an independent, schema-validated claim
about the ticker. A Meta-Agent then reconciles them through a
**deterministic pipeline** — not a search tree, not an LLM "vibes" merge:

1. **Validate** both claims against a strict schema. A malformed claim gets
   one retry with the validation error fed back to the model; a second
   failure marks that agent unavailable and the report proceeds in
   single-agent mode with escalation forced on.
2. **Check for conflict** — a simple, auditable rule: the two agents point
   in opposite directions (bullish vs. bearish) *and* both are confident
   (>0.5). No LLM judgment call here on purpose.
3. **If conflicting, run one critique round** — each agent sees the other's
   claim and evidence and gets one chance to revise its confidence and
   magnitude. Nothing is overwritten; both the original and revised claims
   are kept, so the dissent log always shows what changed and why. This
   replaces the north-star design's Tree-of-Thought/Beam Search — same
   practical benefit (force reconsideration of a strongly opposed view),
   without generating and pruning a tree of hypotheses.
4. **Fit the return distribution** — each claim becomes a normal
   distribution (mean = signed magnitude, std = a function of
   `1 - confidence`), combined into a weighted mixture using a
   regime-conditioned weight table, then summarized into the three-bucket
   distribution. Pure NumPy/SciPy; no LLM ever does this arithmetic.
5. **Generate the rationale** — only now does an LLM write the thesis
   paragraph, and only to narrate the already-fixed numbers.

Every run — both raw agent claims, the critique exchange if any, and the
final synthesis — is persisted before the response returns, so any report
can be reconstructed and audited after the fact.

## Components

### Schemas — the shared contract

`equity_ensemble/schemas/models.py`

The Pydantic models every other component depends on. This is the
interface boundary that let the five parts of this system be built and
tested independently against each other.

| Model | Purpose |
|---|---|
| `Evidence` | `{source, date, snippet}` — the atomic unit of citation. |
| `AgentClaim` | Shared output shape for both specialist agents: `agent`, `ticker`, `direction`, `magnitude_bps`, `confidence`, plus agent-specific fields (`regime_label`/`regime_changed` for Technicals, `sentiment_score`/`key_risk_flags` for Fundamentals/Sentiment), and always non-empty `evidence[]` and `falsifiers[]`. |
| `RetrievalChunk` | A chunk of a filing or news article prepared for vector search: `chunk_id`, `source_type`, `source_date`, `section`, `text`, `embedding`. |
| `Distribution` | `{bullish_pct, neutral_pct, bearish_pct}` — validated to sum to 100. |
| `ForecastReport` | The deliverable. Includes the distribution, recommendation (never "buy"/"sell" — decision-support language only), confidence, escalation flag/reason, thesis, both raw `agent_claims` (the dissent log), `critique_log`, and deduplicated `citations`. |

Falsifiers are worth calling out: every claim is required to state what
observation would *change* it. That's a forcing function against
unfalsifiable hand-waving.

### Technicals Agent

`equity_ensemble/agents/technicals.py`

**Role**: classify the ticker's current price/volume regime and produce a
directional, magnitude-bounded claim.

**How it reasons**: a bounded ReAct loop.

1. Reads its own last-known regime for this ticker from Postgres
   (long-term memory) — so it detects a *transition*, not a fresh
   classification every run.
2. Pulls 2 years of OHLCV plus ADX(14) and SMA(50) from FMP.
3. **Classifies the regime in code, not via the LLM**: ADX ≥ 25 means
   trending (direction read off price-vs-SMA → `Trending Bull` /
   `Trending Bear`); below that, realized volatility distinguishes a quiet
   `Mean-Reverting` tape from a `Volatile/Choppy` one.
4. Asks the LLM to narrate that computed result into a claim — and then
   **overwrites** the LLM's `regime_label`/`regime_changed` fields with the
   code-computed values before returning. The guarantee that "the LLM never
   derives the regime" is enforced, not just requested in the prompt.

Options implied volatility is pulled in as best-effort context (the vendor
endpoint for it is still unconfirmed — see [Known limitations](#known-limitations))
but never blocks the loop if unavailable.

### Fundamentals/Sentiment Agent

`equity_ensemble/agents/fundamentals_sentiment.py`

**Role**: produce a sentiment score and qualitative fundamental read,
grounded in the company's own disclosures rather than the model's
parametric knowledge.

**How it reasons**: sequential tool invocation, in this order:

1. **Fetch recent news** (last 14 days) from FMP and have the LLM score
   each article's sentiment 1–100.
2. **Ingest the latest filing** (10-K/10-Q/8-K text from FMP) — chunk it
   (~500 tokens, 50-token overlap), embed every chunk (Voyage AI), and
   store it in pgvector.
3. **Retrieve the chunks relevant to the claim being formed** — the query
   is built from the most negative headline found in step 1 (so a
   deteriorating narrative gets checked against the actual filing text),
   embedded, searched via cosine-similarity ANN in Postgres, and re-ranked
   by `cosine_similarity × recency_decay(source_date)` (top 5 max). This is
   what lets a genuine disclosure — a guidance cut, an executive departure
   — override a purely price-momentum-driven initial read, and it's what
   keeps the agent from confabulating filing content it never saw.
4. **Produce the final claim**, grounded only in the news scores and
   retrieved chunks it was actually shown. Every `evidence[]` entry must
   cite a specific source; the schema enforces non-empty evidence, though
   only a manual groundedness spot-check (see [Eval](#eval)) actually
   verifies the citations are accurate.

### Meta-Agent

`equity_ensemble/agents/meta_agent.py`

**Role**: reconcile the two specialist claims into one auditable report.
Never places or sizes a trade. This is the piece described in
[How it works](#how-it-works) above — see that section for the full
five-step pipeline (validate → conflict-check → critique → fit → narrate).

Two supporting pieces worth knowing about:

- **Regime-conditioned weighting** (`config/regime_weights.yaml`): a
  static, checked-in lookup table keyed by the Technicals agent's regime
  label. `Volatile/Choppy` regimes weight Technicals 0.6 / Fundamentals
  0.4; trending or steady regimes invert that (0.4 / 0.6), on the
  reasoning that a choppy tape makes price action itself more informative,
  while a stable trend leaves more room for fundamentals to move the
  needle. It's a deliberate simplification — a future phase would learn
  these weights from the backtest log instead.
- **`agents/base.py`**: the shared "validate → on failure, retry once with
  the error appended to context → on second failure, raise
  `AgentUnavailable`" helper, used by both specialist agents and the
  critique round so this policy lives in exactly one place.

### LLM client

`equity_ensemble/llm/client.py`

Every agent and the Meta-Agent call through one interface —
`LLMClient.complete_structured(system_prompt, user_prompt, response_model)`
— never an OpenRouter or Anthropic SDK directly. `OpenRouterClient` is the
real implementation (schema-constrained JSON output, with an optional
fallback model if the primary fails). Swapping providers, or falling back
to a direct Anthropic call, is confined to this one file.

`FakeLLMClient` is a scripted test double used throughout the test suite —
no agent test ever makes a network call.

### Data layer

`equity_ensemble/data/fmp_client.py`, `equity_ensemble/data/retrieval.py`

- **`FMPClient`**: the single wrapper around every Financial Modeling Prep
  endpoint used — profile lookup, OHLCV, technical indicators, filings,
  news, estimate revisions. `RecordedFMPClient` replays JSON fixtures
  (`tests/fixtures/fmp/`) so the whole test suite runs offline.
- **`retrieval.py`**: chunking, embedding (`VoyageEmbedder` for real,
  `FakeEmbedder` — deterministic, hash-seeded, no network — for tests), and
  the recency-weighted ranking function described above.

### Persistence

`equity_ensemble/persistence/db.py`, `equity_ensemble/persistence/migrate.py`

`PostgresDatabase` (real, `asyncpg` + `pgvector`) and `FakeDatabase`
(in-memory) implement the same `Database` protocol: regime history (long-
term agent memory), filing chunks with vector search, and the full
forecast audit trail — including a `list_pending_evaluations` /
`record_realized_outcome` pair used by the eval job below.

Schema (`persistence/migrations/0001_init.sql`, applied by the plain-SQL
`migrate.py` runner — no Alembic, the schema is small and fixed):

| Table | Holds |
|---|---|
| `regime_history` | `(ticker, as_of, regime_label)` — Technicals' long-term memory. |
| `filing_chunks` | Chunked filing/news text + `vector(1024)` embeddings, with a cosine-distance ANN index. |
| `forecasts` | The full `ForecastReport` as JSONB, plus `realized_return`/`brier_score` columns filled in later by the eval job. |

### Graph

`equity_ensemble/graph/build_graph.py`, `equity_ensemble/graph/wiring.py`

A LangGraph `StateGraph`: `START` fans out to both specialist nodes (which
LangGraph runs concurrently since they share a source), both feed a single
`synthesize` node that calls `MetaAgent.run`, then `END`. If a specialist
raises `AgentUnavailable`, its node stores `None` instead of crashing the
run — the Meta-Agent already knows how to handle a missing claim. If *both*
fail, the Meta-Agent's `ValueError` propagates out for the API/CLI layer to
turn into an error response.

`wiring.py` builds the real production stack (OpenRouter + FMP + Postgres +
Voyage) in one place, so the API and CLI are guaranteed to run the
identical graph.

> **Note on the PRD's graph topology**: the original spec describes a
> graph-level "conditional edge for the critique loop." Here, the critique
> loop is a deterministic method on `MetaAgent` (not a separate graph node)
> — re-exploding it into more graph edges would just relocate the same
> logic without changing behavior, so `synthesize` calls it as one step.

### Rendering

`equity_ensemble/render/markdown_report.py`

A pure function, `ForecastReport → str`. Produces: header (ticker, horizon,
timestamp, recommendation, confidence) → probability distribution table →
thesis → dissent log (each agent's claim, plus the critique exchange if one
ran) → citations → a fixed disclaimer footer, printed **unconditionally**
on every single report:

> Decision support only — not a trade recommendation. Requires human review
> before any action.

### API and CLI

`equity_ensemble/api/main.py`, `equity_ensemble/cli/main.py`

Both are thin wrappers over the exact same `graph.build_graph.run_forecast`
call, so their output can never diverge.

- **API**: `POST /forecast {"ticker": "AAPL", "horizon_days": 21}` → `200`
  with the full `ForecastReport` JSON plus a `report_markdown` string.
  Returns `404` for an unknown/unsupported ticker.
- **CLI**: `forecast run AAPL --horizon 21` — writes
  `forecast_AAPL_<run_id>.md` and `.json` to the working directory and
  prints the Markdown to stdout.

Both validate the ticker via an FMP profile lookup *before* the graph fans
out, so an unknown ticker fails fast instead of burning two agent calls.

### Eval

`equity_ensemble/eval/brier_score.py`

A scheduled job, not part of the request path: for every forecast whose
horizon has elapsed with no realized outcome recorded, it pulls the
realized closing price, computes the return, buckets it
(bullish/neutral/bearish, using the same ±50bps neutral band the Meta-Agent
uses when fitting its own distribution), scores it against the stored
forecast with a multi-category Brier score, and writes the outcome back.
A uniform random guess scores ~0.33 — that's the baseline the ensemble is
meant to beat.

## Guardrails

| Guardrail | Mechanism |
|---|---|
| Schema enforcement | Every agent/Meta-Agent output is Pydantic-validated; malformed output retries once, then marks the agent unavailable — never a silent pass-through. |
| No execution surface | No tool anywhere in the system can place, size, or route an order. Enforced by simply never implementing one, not a runtime policy check. |
| Dissent surfaced, not averaged | `ForecastReport.agent_claims` always carries both raw claims; the critique log never overwrites the originals. |
| Human escalation | `escalate_to_analyst`, computed deterministically — never an LLM's call — on low confidence or unresolved post-critique disagreement. |
| Full audit trail | Every run (both raw claims, critique exchange, final synthesis) is persisted before the response returns. |
| Source-dated retrieval | Every filing/news citation carries its own date; retrieval ranking actively decays stale sources. |
| No autonomous computation by the LLM | Regime classification, conflict detection, distribution fitting, and Brier scoring are all plain code — the LLM only narrates already-fixed results. |

## Running it

### Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- Docker (for local Postgres+pgvector) — optional; most of the test suite
  and every fake-backed workflow run without it
- API keys for OpenRouter, Financial Modeling Prep, and Voyage AI if you
  want to hit live vendors (none are required to run the test suite)

### Install

```bash
uv sync --extra dev
```

### Configure

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY, FMP_API_KEY, VOYAGE_API_KEY for live runs
```

`.env.example` documents every variable — see
[Configuration reference](#configuration-reference) below.

### Start Postgres and apply migrations

```bash
docker compose up -d
uv run --env-file .env python -m equity_ensemble.persistence.migrate
```

This starts Postgres+pgvector on `localhost:5433` (chosen to avoid
colliding with a default local Postgres on 5432 — see `docker-compose.yml`
and `.env.example`).

> **Note**: nothing in the app auto-loads `.env` — pass `--env-file .env`
> to every `uv run` invocation below (or `export $(cat .env | xargs)`
> first), otherwise the process reads empty env vars and fails with a
> `KeyError` on first use.

### Run the CLI

```bash
uv run --env-file .env python -m equity_ensemble.cli.main run AAPL --horizon 21
```

Writes `forecast_AAPL_<run_id>.md` and `.json` to the current directory and
prints the report to stdout.

### Run the API

```bash
uv run --env-file .env uvicorn equity_ensemble.api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"ticker": "AAPL", "horizon_days": 21}'
```

Interactive docs are served at `http://127.0.0.1:8000/docs`.

## Testing

```bash
uv run pytest              # full suite
uv run ruff check .        # lint
```

The suite runs entirely against fakes and recorded fixtures by default —
`FakeLLMClient`, `RecordedFMPClient`, `FakeEmbedder`, `FakeDatabase` — so no
API keys or running services are required. Tests in
`tests/integration/test_postgres_db.py` that need a real database
auto-skip if `docker compose up -d` + the migration step above haven't been
run; bring the container up first to exercise them (they cover ANN vector
search ordering and full forecast round-trips against real Postgres).

## Repository layout

```
equity_ensemble/
  llm/client.py               # LLMClient interface + OpenRouter implementation
  agents/
    base.py                   # shared validate-retry helper
    technicals.py              # Technicals Agent
    fundamentals_sentiment.py  # Fundamentals/Sentiment Agent
    meta_agent.py              # Meta-Agent synthesis pipeline
  data/
    fmp_client.py              # FMP API wrapper
    retrieval.py                # chunking, embedding, ranking
  schemas/models.py            # AgentClaim, RetrievalChunk, ForecastReport
  graph/
    build_graph.py             # LangGraph state graph
    wiring.py                   # production dependency wiring
  persistence/
    db.py                      # Postgres + in-memory implementations
    migrate.py                  # SQL migration runner
    migrations/                 # numbered .sql files
  eval/brier_score.py          # calibration scoring job
  render/markdown_report.py    # ForecastReport -> Markdown
  api/main.py                   # FastAPI app
  cli/main.py                   # Typer CLI
  config/regime_weights.yaml    # static regime -> agent weight table
tests/
  unit/                         # pure-function tests, no I/O
  integration/                  # fixture/fake-backed and real-Postgres tests
  fixtures/                     # recorded FMP responses, sample filing text
docker-compose.yml              # local Postgres+pgvector for dev
```

## Configuration reference

All variables live in `.env.example`:

| Variable | Used by | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `llm/client.py` | LLM gateway for every agent + Meta-Agent call. |
| `OPENROUTER_MODEL` | `llm/client.py` | Default model slug — a config value, not a code change. |
| `OPENROUTER_FALLBACK_MODEL` | `llm/client.py` | Optional; retried if the primary model/provider errors out. |
| `FMP_API_KEY` | `data/fmp_client.py` | Financial Modeling Prep — market data, filings, news. |
| `VOYAGE_API_KEY` | `data/retrieval.py` | Embeddings for filing/news retrieval. |
| `DATABASE_URL` | `persistence/db.py` | Postgres connection string; must have the `vector` extension enabled. |

## Known limitations

These are documented, accepted gaps rather than oversights — worth knowing
before extending the system:

- **Options implied volatility**: the real FMP endpoint for it is
  unconfirmed; the Technicals agent uses it as best-effort context only and
  never blocks on it being unavailable.
- **SEC EDGAR fallback**: when FMP doesn't mirror a filing, the real FMP
  client raises `NotImplementedError` with a clear message rather than
  guessing at an unverified EDGAR integration — implement this when a real
  vendor gap is actually hit.
- **Chunking uses word count, not a real tokenizer**: a reasonable MVP
  approximation of "500 tokens, 50 overlap" that avoids an extra
  dependency; swap in a real tokenizer if chunk-size precision starts to
  matter.
- **Regime weights are a static, checked-in table**, not learned — expected
  to mis-price some regimes until a future phase learns weights from the
  backtest log.
- **Horizon-elapsed check in the eval job treats the horizon as calendar
  days**, not trading days — a simplification that's fine until real
  latencies make the distinction matter.
- **No authentication on the API** — add before exposing outside a trusted
  internal network.
