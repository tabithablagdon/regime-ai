/**
 * Distinct from a ticker-not-found error (which stays inline near the
 * input): this is for the 503 case — the reasoning pipeline itself failed
 * or is unreachable (e.g. the LLM gateway isn't hooked up to a real key
 * yet). Deliberately calm, not alarming — this is an expected state during
 * setup, not a crash.
 */
export function EngineUnavailableBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-lg border-l-4 p-4"
      style={{ borderColor: "var(--bearish)", backgroundColor: "var(--surface)" }}
    >
      <p className="font-semibold">Analysis engine unavailable</p>
      <p className="mt-1 text-sm" style={{ color: "var(--ink-secondary)" }}>
        {message}
      </p>
    </div>
  );
}
