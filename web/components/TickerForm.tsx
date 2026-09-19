"use client";

import { useState } from "react";

export function TickerForm({
  onSubmit,
  disabled,
  fieldError,
}: {
  onSubmit: (ticker: string, horizonDays: number) => void;
  disabled: boolean;
  fieldError: string | null;
}) {
  const [ticker, setTicker] = useState("");
  const [horizonDays, setHorizonDays] = useState(21);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (ticker.trim()) onSubmit(ticker.trim().toUpperCase(), horizonDays);
      }}
      className="flex flex-wrap items-end gap-3"
    >
      <div className="flex flex-col gap-1">
        <label htmlFor="ticker" className="text-sm font-medium">
          Ticker symbol
        </label>
        <input
          id="ticker"
          name="ticker"
          type="text"
          required
          autoComplete="off"
          autoCapitalize="characters"
          placeholder="AAPL"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          disabled={disabled}
          className="w-36 rounded-md border px-3 py-2 uppercase focus:outline-none focus:ring-2"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface)" }}
        />
      </div>

      <div className="flex flex-col gap-1">
        <label htmlFor="horizon" className="text-sm font-medium">
          Horizon (days)
        </label>
        <input
          id="horizon"
          name="horizon"
          type="number"
          min={1}
          max={252}
          value={horizonDays}
          onChange={(e) => setHorizonDays(Number(e.target.value))}
          disabled={disabled}
          className="w-24 rounded-md border px-3 py-2 focus:outline-none focus:ring-2"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface)" }}
        />
      </div>

      <button
        type="submit"
        disabled={disabled || !ticker.trim()}
        className="rounded-md px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        style={{ backgroundColor: "var(--bullish)" }}
      >
        {disabled ? "Analyzing…" : "Get forecast"}
      </button>

      {fieldError && (
        <p className="w-full text-sm font-medium" style={{ color: "var(--bearish)" }}>
          {fieldError}
        </p>
      )}
    </form>
  );
}
