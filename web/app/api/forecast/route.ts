import { NextResponse } from "next/server";
import type { ForecastErrorBody, ForecastReport } from "@/lib/types";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const ENGINE_UNAVAILABLE_MESSAGE =
  "The analysis engine is not available right now. Please try again later.";

/**
 * Proxies to the FastAPI backend's POST /forecast, server-side, so the
 * browser never talks to the backend directly (no CORS config needed, the
 * backend's address stays out of client bundles). This is also the one
 * place that turns backend failure modes into the three UI-facing outcomes:
 * ticker not found (404), analysis engine unavailable (503, including an
 * unreachable backend), or a successful report (200).
 */
export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const ticker = typeof body?.ticker === "string" ? body.ticker.trim() : "";
  const horizonDays =
    typeof body?.horizon_days === "number" ? body.horizon_days : 21;

  if (!ticker) {
    return NextResponse.json<ForecastErrorBody>(
      { kind: "not_found", message: "Enter a ticker symbol." },
      { status: 400 },
    );
  }

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/forecast`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ticker, horizon_days: horizonDays }),
      // A forecast can legitimately take up to ~90s (PRD P95 target); give
      // it real room rather than the framework's default fetch timeout.
      signal: AbortSignal.timeout(120_000),
    });
  } catch {
    // Backend unreachable entirely (not running, wrong URL, timed out) is
    // the same story to the user as the backend replying with a failure:
    // the analysis engine isn't available.
    return NextResponse.json<ForecastErrorBody>(
      { kind: "engine_unavailable", message: ENGINE_UNAVAILABLE_MESSAGE },
      { status: 503 },
    );
  }

  if (backendResponse.status === 404) {
    const detail = await backendResponse
      .json()
      .then((b) => b.detail as string)
      .catch(() => `Unknown or unsupported ticker: ${ticker}`);
    return NextResponse.json<ForecastErrorBody>(
      { kind: "not_found", message: detail },
      { status: 404 },
    );
  }

  if (!backendResponse.ok) {
    return NextResponse.json<ForecastErrorBody>(
      { kind: "engine_unavailable", message: ENGINE_UNAVAILABLE_MESSAGE },
      { status: 503 },
    );
  }

  const report = (await backendResponse.json()) as ForecastReport;
  return NextResponse.json(report, { status: 200 });
}
