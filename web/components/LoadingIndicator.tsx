"use client";

import { useEffect, useState } from "react";

/**
 * Stage copy timed against real measured latency (see the backend's
 * LangSmith traces), not guessed: Technicals finishes in ~8-12s, but the
 * two specialists run concurrently and Fundamentals/Sentiment (filing
 * retrieval + embedding) dominates at ~60s, then the Meta-Agent's
 * critique/rationale step adds ~15-25s more. This can't reflect the
 * backend's *actual* progress (there's no streaming/progress endpoint —
 * it's one blocking POST /forecast), so it's an honest approximation
 * rather than a real progress bar, framed with elapsed time so it never
 * overclaims precision.
 */
const STAGES: { afterSeconds: number; text: string }[] = [
  { afterSeconds: 0, text: "Fetching market data and technical indicators…" },
  { afterSeconds: 8, text: "Reading recent filings and news…" },
  { afterSeconds: 65, text: "Reconciling both agents' claims…" },
  { afterSeconds: 85, text: "Still working — this one's taking a bit longer than usual…" },
];

function currentStageText(elapsedSeconds: number): string {
  let text = STAGES[0].text;
  for (const stage of STAGES) {
    if (elapsedSeconds >= stage.afterSeconds) text = stage.text;
  }
  return text;
}

export function LoadingIndicator() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <span
        className="h-6 w-6 shrink-0 animate-spin-ring rounded-full border-[3px]"
        style={{ borderColor: "var(--neutral)", borderTopColor: "var(--bullish)" }}
        aria-hidden
      />
      <div>
        <p>{currentStageText(elapsed)}</p>
        <p className="text-sm" style={{ color: "var(--ink-muted)" }}>
          {elapsed}s elapsed — both specialist agents run concurrently, then get
          reconciled by the Meta-Agent.
        </p>
      </div>
    </div>
  );
}
