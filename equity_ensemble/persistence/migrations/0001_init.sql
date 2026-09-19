-- Initial schema (PRD §13). Applied by a small migrate.py runner (Track E),
-- not Alembic — MVP scope doesn't need schema-diffing.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS regime_history (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    as_of DATE NOT NULL,
    regime_label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_regime_history_ticker_date ON regime_history (ticker, as_of DESC);

-- Embedding dimension MUST match the chosen Voyage model (PRD §16 risk) —
-- confirm before running this migration. voyage-3 / voyage-finance-2 = 1024.
CREATE TABLE IF NOT EXISTS filing_chunks (
    chunk_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('10-K', '10-Q', '8-K', 'news')),
    source_date DATE NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_filing_chunks_ticker ON filing_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_filing_chunks_embedding ON filing_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id UUID PRIMARY KEY,
    ticker TEXT NOT NULL,
    horizon_days INT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    report JSONB NOT NULL,              -- full ForecastReport, incl. agent_claims + critique_log
    realized_return DOUBLE PRECISION,   -- filled in by eval/brier_score.py once horizon elapses
    brier_score DOUBLE PRECISION,
    llm_token_usage JSONB,              -- cost/latency telemetry, PRD §11
    vendor_call_counts JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_forecasts_ticker_date ON forecasts (ticker, generated_at DESC);
