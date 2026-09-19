# Final Capstone Report: Regime AI, a Multi-Agent Equity Forecast Ensemble

**Student:** Tabitha Blagdon
**Repository:** https://github.com/tabithablagdon/regime-ai
**Date:** September 19, 2026

---

## 1. Problem and intended user

Equity forecasting is a multi-modal problem. The signal lives in numeric time
series, regulatory filings, and unstructured news at the same time, and the
correct weighting of those sources changes with the market regime. Momentum
dominates in trending tapes, fundamentals reassert around earnings, and a
single confident number hides all of that. Individual signals routinely
conflict: strong fundamentals against weak price momentum is the normal case,
not the exception.

Human research teams handle this by convening specialists and arguing.
Regime AI automates that structure. It runs independent specialist agents over
different evidence bases, forces them to reconcile, and returns a probability
distribution instead of a point estimate, so uncertainty is a first-class
output rather than an afterthought.

**Primary user:** a buy-side research analyst or quantitative portfolio manager
who wants a fast, auditable second opinion on a single ticker before sizing a
position. They read the report, not the code, so the report has to stand on its
own.

**Secondary user:** a risk or research manager who needs to see why a view was
formed, where the two agents disagreed, and which agent to hold accountable
when the view turns out to be wrong.

**Why it matters:** a single general-purpose model cannot do this job
credibly. It has no live prices or filings, it produces confident prose rather
than calibrated probabilities, it makes arithmetic errors on volatility and
weighting, and it collapses four evidence bases into whichever narrative it
reads first. Most importantly for a professional user, a monolithic answer
offers no decomposition, so a number cannot be traced to a source.

## 2. Final system architecture

Two specialist agents run concurrently and produce independent, schema-validated
claims. A Meta-Agent reconciles them through a deterministic pipeline and emits
one auditable report.

```
Client (CLI, POST /forecast, or the Next.js web UI)
        |
        v
Orchestrator (FastAPI / Typer): validates the ticker via an FMP profile lookup
        |
        v
LangGraph state graph
        |
        +----------------------+
        v                      v
  Technicals Agent      Fundamentals/Sentiment Agent     (run concurrently)
        |                      |
        +----------+-----------+
                   v
        Meta-Agent (synthesize node)
   validate -> conflict check -> conditional single critique round
   -> fit distribution -> generate rationale
                   |
                   v
   ForecastReport -> Postgres audit trail -> Markdown + JSON
```

**Technicals Agent.** A bounded ReAct loop. It reads its own last-known regime
label for the ticker from Postgres (long-term memory), pulls OHLCV plus ADX(14)
and SMA(50) from FMP, then classifies the regime in code: ADX at or above 25
means trending, with direction read from price versus the moving average, and
below that threshold realized volatility separates a quiet Mean-Reverting tape
from a Volatile/Choppy one. The LLM only narrates that computed result, and the
agent overwrites the model's `regime_label` and `regime_changed` fields with the
code-computed values before returning. The guarantee is enforced, not requested
in a prompt.

**Fundamentals/Sentiment Agent.** Sequential tool invocation. It fetches the
last 14 days of news and scores each article 1 to 100, ingests the latest
filing text, chunks it, embeds every chunk with Voyage AI, and stores the
vectors in pgvector. It then builds a retrieval query from the most negative
headline, searches by cosine similarity, and re-ranks by
`cosine_similarity * recency_decay(source_date)`. Only then does it form its
claim, grounded in the news scores and filing chunks it was actually shown.
This is what lets a real disclosure override a momentum-driven read, and what
keeps the agent from confabulating filing content.

**Meta-Agent.** A deterministic five-step pipeline, not a search tree:

1. **Validate** each claim against its Pydantic schema. One failure triggers a
   retry with the validation error appended to context; a second failure marks
   that agent unavailable and the run continues in single-agent mode with
   escalation forced on.
2. **Conflict check** using an inspectable rule: directions are opposed
   (bullish against bearish, not neutral against either) and both confidences
   exceed 0.5.
3. **Single critique round**, only if that rule fires. Each specialist sees the
   other's claim and evidence and may revise confidence and magnitude. Nothing
   is overwritten, so the dissent log shows both the original and revised
   claims.
4. **Fit the distribution.** Each claim becomes a normal distribution with mean
   equal to the signed magnitude and standard deviation a function of
   `1 - confidence`. The two are combined as a weighted mixture using a
   regime-conditioned weight table and summarized by Monte Carlo sampling into
   bullish, neutral, and bearish buckets around a 50 bps neutral band. Pure
   NumPy, no LLM arithmetic.
5. **Generate the rationale** last, after every numeric field is fixed, so the
   model narrates the computed result instead of inventing it.

**Supporting components.** A single `LLMClient` interface that every agent calls
through, so the provider is a one-file change. A single FMP wrapper. Postgres
holds three tables: `regime_history` for agent memory, `filing_chunks` for
vector search, and `forecasts` for the full audit trail. A pure rendering
function turns a `ForecastReport` into Markdown. A scheduled Brier scoring job
sits outside the request path. Every agent emits structured audit logs on call,
intermediate reasoning, and final decision.

**Integration guarantee.** The API, the CLI, and the web UI are thin wrappers
over the same `run_forecast` call, and `wiring.py` builds the production
dependency stack in one place, so the three surfaces cannot diverge.

## 3. System goal, scope, design rationale, and constraints

**Goal.** Given one U.S.-listed equity ticker and a horizon (21 trading days by
default), produce a complete, auditable forecast report without manual
intervention. Successful performance means: the report always shows both raw
agent claims rather than only the blended result, every fundamental or
sentiment claim cites a specific dated source, ambiguous cases are flagged for
a human instead of answered confidently, and the output is available as both
human-readable Markdown and machine-readable JSON.

**In scope.** Single ticker, single on-demand query. Two specialists plus a
Meta-Agent. Retrieval over the current filing set. Postgres persistence and the
calibration scoring job.

**Out of scope, deliberately.** Watchlists, scheduled monitoring, multi-ticker
batching, authentication, and high availability. This is not a production
trading system.

**Design rationale.** The north-star design called for four specialists,
Tree-of-Thought with beam search, a managed vector database, and a full
backtesting harness. That is a sound end state but not a buildable first
release, so each cut was made against a stated reason:

| Decision | Rationale |
|---|---|
| Two specialists instead of four | Technicals and Fundamentals/Sentiment are both ticker-specific and already force genuine disagreement between numeric and qualitative evidence. A macro read applies nearly uniformly across tickers, and reliable positioning data is the most fragmented and expensive part of the data stack. |
| Deterministic pipeline instead of Tree-of-Thought | With two inputs a search tree collapses to a handful of trivial branches. The conflict check plus one critique round preserves the property that actually matters, which is that dissent is surfaced rather than averaged away, and removes most of the latency and token cost. |
| Postgres with pgvector instead of a managed vector store | At one ticker of filings per query, pgvector does the job with no extra vendor, and it lives in the same database as the forecasts and regime history. |
| Brier scoring instead of Backtrader | The system never places a trade, so most of a trading simulator is unused surface. The real question is whether the distribution was calibrated, which is a short scoring job. |
| Regime classification, conflict detection, and distribution fitting in code | Tokens sampled from a language model are not arithmetic. Every number is computed and only then narrated. |
| One `LLMClient` interface | Prompt-caching pass-through and structured-output fidelity through a third-party router were unverified, so the mitigation was architectural: nothing else in the codebase talks to the gateway's API shape. |

**Constraints.**

- No code path can place, size, or route a trade. This is enforced by never
  implementing such a tool, not by a runtime policy check.
- The LLM never computes a number that reaches the report.
- Recommendation language is decision-support only. The vocabulary is
  `bullish_lean`, `neutral`, `bearish_lean`, and `requires_review`, never buy
  or sell.
- A fixed disclaimer is printed on every report, unconditionally.
- P95 latency target of 90 seconds per query.
- The test suite must run with no API keys and no running services.

## 4. Evaluation criteria

| Criterion | Target |
|---|---|
| Forecast calibration | Multi-category Brier score on elapsed forecasts, beating the roughly 0.33 uniform-random baseline. |
| Groundedness | Every `evidence[]` entry in a sampled report checks out by hand against the source filing or article. |
| Functional completeness | Each of the eight build milestones meets its definition of done, verified by an automated test. |
| Escalation correctness | Low confidence or unresolved post-critique disagreement always produces `requires_review`, never a confident recommendation. |
| Determinism of the numeric path | Regime classification, conflict detection, and distribution fitting are unit-testable as pure functions and produce identical output for identical input. |
| Graceful degradation | One failed specialist yields a single-agent report with escalation set, not a crash. |
| Latency and cost | P95 at or under 90 seconds, with per-run token and vendor call counts recorded. |

## 5. Design evolution across the program

The design moved through three stages: the north-star proposal, a scoping pass
that turned it into a buildable specification, and the build itself. Mapped to
the course concepts:

| Concept | Early design | Final system | Why it changed |
|---|---|---|---|
| Tool calling | Four agents over four vendor stacks, including a market-data API that turned out not to exist under the name used | One FMP wrapper covering profile, OHLCV, indicators, filings, and news, behind an interface with a fixture-replaying twin | Single-vendor integration cut the surface area, and recorded fixtures made the whole suite runnable offline |
| Reasoning and chain of thought | ReAct loops described per agent | Bounded ReAct in Technicals with the regime classifier moved into code; sequential tool invocation in Fundamentals/Sentiment | Letting the model derive the regime reintroduced exactly the arithmetic and confabulation risk the project exists to avoid, so the loop narrates a computed result |
| Knowledge and memory | Full RAG stack on a managed vector database | pgvector with recency-decayed re-ranking, plus a `regime_history` table as long-term agent memory | Retrieval was kept because it is the defense against confabulated filing content; only the managed vendor was cut. Regime memory turned a fresh classification into transition detection |
| Further reasoning | Tree-of-Thought with beam search across four agents | Conflict check plus exactly one critique round, with both pre- and post-critique claims retained | The highest-leverage simplification. It keeps the safety property and drops the search-tree cost, and it bounds latency by construction |
| Multi-agent coordination | Four specialists and a meta-agent, with a graph-level conditional critique edge | LangGraph fan-out to two specialists into one synthesize node, with the critique loop as a deterministic method | Re-exploding the critique loop into graph edges would relocate the same logic without changing behavior |
| Safety | Guardrail classifier models and adversarial red-team CI | Pydantic schema enforcement with retry, deterministic escalation, no execution surface, full audit trail, unconditional disclaimer | The cheap, high-value guardrails were kept. The classifier layer is deferred until a free-text user input surface exists, since the only untrusted text today is fetched by our own agents from a fixed vendor |

Three refinements came after the specification, during the build:

1. **Ticker validation moved ahead of the fan-out**, so an unknown ticker fails
   fast instead of burning two agent calls.
2. **A clean 503 replaced a raw 500** when the reasoning pipeline fails. The
   real cause is logged server-side and the client sees a calm message instead
   of a stack trace.
3. **Structured audit logging was added to every agent**, recording the call,
   the intermediate reasoning, and the final decision, so a run can be
   reconstructed from logs and not only from the database.

A small Next.js web UI was also added beyond the original MVP scope, which
called for CLI and API only.

## 6. Implementation approach

Python 3.11 or newer, managed with `uv`.

| Layer | Choice | Role |
|---|---|---|
| Orchestration | LangGraph | Explicit state graph, concurrent fan-out, single join node |
| LLM gateway | OpenRouter, Claude Sonnet class by default, optional fallback model | One interface for every reasoning call; model choice is config, not code |
| Data | Financial Modeling Prep | Profile, OHLCV, ADX and SMA, filings, news, estimate revisions |
| Embeddings | Voyage AI | Finance-domain embeddings for filing and news retrieval |
| Storage | Postgres with pgvector, via asyncpg | Forecast audit trail, regime memory, vector search |
| Schemas | Pydantic v2 | Every agent and Meta-Agent output; failure drives the retry path |
| Numerics | NumPy | Distribution fitting by Monte Carlo sampling, Brier scoring |
| Surfaces | FastAPI, Typer, Next.js | Three thin wrappers over one `run_forecast` call |
| Quality | pytest, ruff | 111 tests, all offline |

Two implementation decisions carried most of the weight. First, the Pydantic
schemas were written before anything else, which made the five parts of the
system independently buildable and testable against one contract. Second,
every external dependency has a paired test double: `FakeLLMClient`,
`RecordedFMPClient`, `FakeEmbedder`, and `FakeDatabase`. The full suite
therefore runs with no keys and no services, and Postgres-dependent tests skip
automatically when the container is not up.

## 7. Evaluation results, strengths, and limitations

**How it was evaluated.** Each milestone has a corresponding automated test
that encodes its definition of done, run against recorded FMP fixtures and
scripted model responses so results are deterministic. Pure functions, the
regime classifier, the conflict rule, distribution fitting, Brier scoring, and
retrieval ranking, are unit-tested directly. A live end-to-end run was then
attempted against real vendors.

**Results.**

- 111 tests pass, 84 unit and 27 integration, in about two seconds with no
  network access.
- The API and the CLI produce byte-identical Markdown and matching JSON for the
  same ticker and graph, which is the parity property the two surfaces exist to
  guarantee.
- A deliberately conflicting pair of claims triggers exactly one critique round
  and records four entries in the dissent log, with the original claims intact.
- An unresolved conflict after critique, and a single unavailable agent, both
  produce `requires_review` with a populated escalation reason.
- Both specialists failing propagates a clear error rather than emitting an
  empty report.
- The scoring job scores an aged forecast, skips one that has not reached its
  horizon, and does not rescore an already-scored one.
- A live run against real keys exercised the FMP data path and a real
  structured OpenRouter call successfully, then failed at filing embedding
  because the Voyage account is on a free tier limited to 3 requests and 10,000
  tokens per minute. The API returned a clean 503 with the cause logged
  server-side.

**Strengths.** The auditability is real rather than asserted: every number in
the report is produced by inspectable code, both raw claims are always carried,
every fundamental claim is dated and cited, and the whole run is persisted
before the response returns. Escalation is deterministic, so the system cannot
talk itself out of asking for a human. The fake-and-fixture layer means the
reasoning architecture can be verified without spending a cent on vendors.

**Limitations.**

- **No live calibration number yet.** The Brier job is implemented and tested
  against synthetic aged forecasts, but no real forecast has reached its
  horizon, so the claim of beating the 0.33 baseline is untested empirically.
  This is the most significant gap.
- **Graceful degradation covers schema failures, not vendor failures.** The
  graph catches `AgentUnavailable` and continues in single-agent mode, but a
  vendor exception such as the Voyage rate limit propagates and fails the whole
  run. The specification asks for single-agent degradation when a vendor is
  unreachable, so this needs the specialist nodes to catch vendor errors too.
- **Untrusted-content tagging is not implemented.** Retrieved filing and news
  text is passed to the model as formatted lines, not wrapped in explicit inert
  data delimiters, and the system prompt instructs grounding without instructing
  the model to treat retrieved text as passive data. The prompt-injection
  surface is narrow, since all text is fetched by our own agents from one
  vendor, but this is a specified guardrail that is still open.
- **Cost and latency telemetry is schema-only.** The `forecasts` table has an
  `llm_token_usage` column and nothing writes it, so the 90 second P95 target
  is unmeasured.
- **Regime weights are static**, a checked-in table rather than learned from
  outcomes, and will misprice some regimes.
- **Chunking counts words, not tokens**, and the horizon-elapsed check in the
  scoring job treats the horizon as calendar days rather than trading days.
- **Groundedness checking is manual**, with no automated evaluation harness.

## 8. Safety, reliability, and human oversight

| Concern | Mechanism |
|---|---|
| Unauthorized execution | No tool anywhere in the system can place, size, or route an order, enforced by never implementing one. There is no runtime policy check to misconfigure. |
| Overconfident output | `escalate_to_analyst` is computed deterministically, never by the model, whenever overall confidence falls below 0.4 or the agents still disagree after the critique round. It forces `requires_review` language. |
| Hidden disagreement | `agent_claims` always carries both raw claims and the critique log never overwrites the originals, so a blended distribution cannot paper over a real split. |
| Hallucinated numbers | Regime classification, conflict detection, distribution fitting, and Brier scoring are all plain code. The model narrates fixed results. |
| Hallucinated sources | Claims are formed only from retrieved chunks and scored headlines, the schema requires non-empty evidence, and every citation carries its own date. |
| Unfalsifiable claims | Each claim must state the observation that would change it. |
| Malformed output | Pydantic validation on every output, one retry with the error fed back, then the agent is marked unavailable. Never a silent pass-through. |
| Loss of the reasoning trace | The full run is persisted before the response returns, and every agent logs its call, intermediate reasoning, and decision. |
| Stale evidence | Retrieval ranking decays older sources and filing age appears in every citation. |
| Leaked internals | A pipeline failure returns a 503 with a neutral message while the real cause is logged server-side. |

**Human oversight** is structural rather than advisory. The report is a
recommendation to a person, the disclaimer stating that human review is
required appears on every report unconditionally, the escalation flag routes
ambiguous cases to an analyst by construction, and the dissent log gives a risk
manager enough to attribute a wrong view to a specific agent and a specific
piece of evidence.

**Known unintended actions the system could take**, and the mitigations: it
could present an overconfident view on an ambiguous setup, which escalation
addresses; it could cite a real source for a claim that source does not
support, which the manual groundedness check addresses and automated evaluation
would address better; it could absorb instructions embedded in a filing
footnote or news article, which is the open untrusted-content tagging gap noted
above; and it could produce a stale view if a vendor silently serves old data,
which dated citations make visible but do not prevent.

## 9. Next steps

1. Run live forecasts on a small ticker set, let the horizon elapse, and report
   a real Brier score against the 0.33 baseline.
2. Catch vendor exceptions in the specialist nodes so a vendor outage degrades
   to single-agent mode instead of failing the run.
3. Wrap retrieved text in explicit inert-data delimiters and instruct the model
   to treat it as passive data.
4. Write token usage and latency per run into the column that already exists,
   then verify the P95 target.
5. Add an automated groundedness evaluation, then learn the regime weights from
   the accumulated scoring log.
6. Add authentication before the API is exposed beyond a trusted network.

## 10. Public GitHub repository

**https://github.com/tabithablagdon/regime-ai**

The repository is public and contains:

- A README covering the problem, architecture, every component's
  responsibility, guardrails, setup, configuration, and known limitations.
- All source under `equity_ensemble/`, plus the Next.js UI under `web/`.
- The test suite under `tests/`, with recorded FMP fixtures and sample filing
  text so it runs offline.
- `docker-compose.yml` and a `Makefile` for one-command local setup.

To review it:

```bash
uv sync --extra dev
uv run pytest          # 111 tests, no API keys or services needed
uv run ruff check .
```

To run it end to end, copy `.env.example` to `.env`, add the OpenRouter, FMP,
and Voyage keys, then:

```bash
make up
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"ticker": "AAPL", "horizon_days": 21}'
```

`make logs` tails the audit trail, and `make down` stops everything.
