// Mirrors equity_ensemble/schemas/models.py exactly — keep in sync with the
// backend if those Pydantic models change.

export type Direction = "bullish" | "neutral" | "bearish";

export type AgentName = "technicals" | "fundamentals_sentiment";

export type Recommendation =
  | "bullish_lean"
  | "neutral"
  | "bearish_lean"
  | "requires_review";

export interface Evidence {
  source: string;
  date: string; // ISO date
  snippet: string;
}

export interface AgentClaim {
  agent: AgentName;
  ticker: string;
  direction: Direction;
  magnitude_bps: number;
  confidence: number; // 0-1
  regime_label: string | null;
  regime_changed: boolean | null;
  sentiment_score: number | null; // 1-100
  key_risk_flags: string[];
  evidence: Evidence[];
  falsifiers: string[];
}

export interface Distribution {
  bullish_pct: number;
  neutral_pct: number;
  bearish_pct: number;
}

export interface ForecastReport {
  run_id: string;
  ticker: string;
  horizon_days: number;
  generated_at: string; // ISO datetime
  distribution: Distribution;
  recommendation: Recommendation;
  overall_confidence: number; // 0-1
  escalate_to_analyst: boolean;
  escalation_reason: string | null;
  thesis: string;
  agent_claims: AgentClaim[];
  critique_log: string[];
  citations: Evidence[];
  report_markdown: string;
}

/** Shape returned by our own /api/forecast route on any non-200 response. */
export interface ForecastErrorBody {
  message: string;
  kind: "not_found" | "engine_unavailable";
}
