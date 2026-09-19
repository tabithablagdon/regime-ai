import type { ForecastReport } from "@/lib/types";
import { AgentClaimCard } from "./AgentClaimCard";
import { DistributionMeter } from "./DistributionMeter";

const RECOMMENDATION_LABEL: Record<ForecastReport["recommendation"], string> = {
  bullish_lean: "Bullish Lean",
  neutral: "Neutral",
  bearish_lean: "Bearish Lean",
  requires_review: "Requires Analyst Review",
};

const DISCLAIMER =
  "Decision support only — not a trade recommendation. Requires human review before any action.";

/**
 * Mirrors render/markdown_report.py's structure exactly (header,
 * distribution, thesis, dissent log incl. critique exchange, citations,
 * unconditional disclaimer) so the web UI and the Markdown/CLI report tell
 * the same story.
 */
export function ForecastReportView({ report }: { report: ForecastReport }) {
  return (
    <div className="space-y-8">
      <header>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-2xl font-bold">
            {report.ticker} — {report.horizon_days}-Day Horizon
          </h2>
          <span className="text-sm" style={{ color: "var(--ink-muted)" }}>
            Generated {new Date(report.generated_at).toLocaleString()}
          </span>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <span className="rounded-full px-3 py-1 text-sm font-semibold" style={{ backgroundColor: "var(--neutral)" }}>
            {RECOMMENDATION_LABEL[report.recommendation]}
          </span>
          <span className="text-sm" style={{ color: "var(--ink-secondary)" }}>
            Overall confidence: {(report.overall_confidence * 100).toFixed(0)}%
          </span>
          {report.trace_url && (
            <a
              href={report.trace_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium underline underline-offset-2"
              style={{ color: "var(--bullish)" }}
            >
              View execution trace ↗
            </a>
          )}
        </div>
        {report.escalate_to_analyst && (
          <div
            className="mt-3 rounded-md border-l-4 p-3 text-sm"
            style={{ borderColor: "var(--bearish)", backgroundColor: "var(--surface)" }}
          >
            <span className="font-semibold">Requires Analyst Review — </span>
            {report.escalation_reason}
          </div>
        )}
      </header>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--ink-muted)" }}>
          Probability Distribution
        </h3>
        <DistributionMeter distribution={report.distribution} />
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--ink-muted)" }}>
          Thesis
        </h3>
        <p className="leading-relaxed">{report.thesis}</p>
      </section>

      <section>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--ink-muted)" }}>
          Dissent Log
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          {report.agent_claims.map((claim) => (
            <AgentClaimCard key={claim.agent} claim={claim} />
          ))}
        </div>

        {report.critique_log.length > 0 && (
          <div className="mt-4 rounded-lg border p-4 text-sm" style={{ borderColor: "var(--border)" }}>
            <p className="mb-2 font-semibold">Critique exchange</p>
            <ul className="space-y-1" style={{ color: "var(--ink-secondary)" }}>
              {report.critique_log.map((entry, i) => (
                <li key={i}>{entry}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--ink-muted)" }}>
          Citations
        </h3>
        {report.citations.length > 0 ? (
          <ul className="space-y-1 text-sm" style={{ color: "var(--ink-secondary)" }}>
            {report.citations.map((c, i) => (
              <li key={i}>
                <span className="font-medium">{c.source}</span> ({c.date}): {c.snippet}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm" style={{ color: "var(--ink-muted)" }}>
            No citations recorded.
          </p>
        )}
      </section>

      <footer
        className="border-t pt-4 text-xs"
        style={{ borderColor: "var(--gridline)", color: "var(--ink-muted)" }}
      >
        {DISCLAIMER}
      </footer>
    </div>
  );
}
