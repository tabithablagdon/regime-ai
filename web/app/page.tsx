"use client";

import { useState } from "react";
import { EngineUnavailableBanner } from "@/components/EngineUnavailableBanner";
import { ForecastReportView } from "@/components/ForecastReportView";
import { TickerForm } from "@/components/TickerForm";
import type { ForecastErrorBody, ForecastReport } from "@/lib/types";

type RequestState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; report: ForecastReport }
  | { status: "error"; error: ForecastErrorBody };

export default function Home() {
  const [state, setState] = useState<RequestState>({ status: "idle" });

  async function handleSubmit(ticker: string, horizonDays: number) {
    setState({ status: "loading" });
    try {
      const res = await fetch("/api/forecast", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ticker, horizon_days: horizonDays }),
      });
      const body = await res.json();
      if (res.ok) {
        setState({ status: "success", report: body as ForecastReport });
      } else {
        setState({ status: "error", error: body as ForecastErrorBody });
      }
    } catch {
      setState({
        status: "error",
        error: {
          kind: "engine_unavailable",
          message: "The analysis engine is not available right now. Please try again later.",
        },
      });
    }
  }

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10 sm:px-6">
      <h1 className="text-3xl font-bold">Regime AI</h1>
      <p className="mt-2" style={{ color: "var(--ink-secondary)" }}>
        Enter a U.S.-listed equity ticker for an auditable, multi-agent
        forecast — decision support only, not a trade recommendation.
      </p>

      <div className="mt-6">
        <TickerForm
          onSubmit={handleSubmit}
          disabled={state.status === "loading"}
          fieldError={
            state.status === "error" && state.error.kind === "not_found"
              ? state.error.message
              : null
          }
        />
      </div>

      <div className="mt-8">
        {state.status === "loading" && (
          <p style={{ color: "var(--ink-muted)" }}>
            Running both specialist agents and reconciling their claims —
            this can take up to a minute or so…
          </p>
        )}

        {state.status === "error" && state.error.kind === "engine_unavailable" && (
          <EngineUnavailableBanner message={state.error.message} />
        )}

        {state.status === "success" && <ForecastReportView report={state.report} />}
      </div>
    </main>
  );
}
