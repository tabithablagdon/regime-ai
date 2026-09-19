import type { AgentClaim } from "@/lib/types";

const AGENT_DISPLAY_NAME: Record<AgentClaim["agent"], string> = {
  technicals: "Technicals",
  fundamentals_sentiment: "Fundamentals / Sentiment",
};

const DIRECTION_COLOR_VAR: Record<AgentClaim["direction"], string> = {
  bullish: "var(--bullish)",
  neutral: "var(--ink-muted)",
  bearish: "var(--bearish)",
};

/**
 * One specialist's raw, unedited claim. Rendered twice per report, side by
 * side — this is what keeps the "both agents' raw claims, never silently
 * averaged" property visible in the UI (mirrors the Dissent Log section of
 * render/markdown_report.py).
 */
export function AgentClaimCard({ claim }: { claim: AgentClaim }) {
  return (
    <div
      className="rounded-lg border p-4"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--surface)" }}
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-semibold">{AGENT_DISPLAY_NAME[claim.agent]}</h3>
        <span
          className="rounded-full px-2 py-0.5 text-xs font-semibold capitalize text-white"
          style={{ backgroundColor: DIRECTION_COLOR_VAR[claim.direction] }}
        >
          {claim.direction}
        </span>
      </div>

      <dl className="mt-3 space-y-1 text-sm" style={{ color: "var(--ink-secondary)" }}>
        <Row label="Magnitude">{claim.magnitude_bps} bps</Row>
        <Row label="Confidence">{(claim.confidence * 100).toFixed(0)}%</Row>
        {claim.regime_label && (
          <Row label="Regime">
            {claim.regime_label}
            {claim.regime_changed ? " (changed)" : ""}
          </Row>
        )}
        {claim.sentiment_score !== null && (
          <Row label="Sentiment score">{claim.sentiment_score}/100</Row>
        )}
      </dl>

      {claim.key_risk_flags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {claim.key_risk_flags.map((flag) => (
            <span
              key={flag}
              className="rounded px-1.5 py-0.5 text-xs"
              style={{ backgroundColor: "var(--neutral)", color: "var(--foreground)" }}
            >
              {flag}
            </span>
          ))}
        </div>
      )}

      <div className="mt-3 text-sm">
        <p className="font-medium" style={{ color: "var(--ink-secondary)" }}>
          Falsifiers
        </p>
        <ul className="mt-1 list-inside list-disc" style={{ color: "var(--ink-muted)" }}>
          {claim.falsifiers.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      </div>

      <div className="mt-3 text-sm">
        <p className="font-medium" style={{ color: "var(--ink-secondary)" }}>
          Evidence
        </p>
        <ul className="mt-1 space-y-1">
          {claim.evidence.map((e, i) => (
            <li key={i} style={{ color: "var(--ink-muted)" }}>
              <span className="font-medium">{e.source}</span> ({e.date}): {e.snippet}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4">
      <dt>{label}</dt>
      <dd className="font-medium" style={{ color: "var(--foreground)" }}>
        {children}
      </dd>
    </div>
  );
}
