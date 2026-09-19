import type { Distribution } from "@/lib/types";

/**
 * Diverging 3-segment meter: bullish <-> neutral <-> bearish. Polarity data
 * (which side of a baseline) gets a diverging palette — two hues + a
 * neutral gray midpoint, not arbitrary categorical colors — per the
 * dataviz skill's color-formula. Segments are separated by a 2px
 * surface-color gap rather than a border (the skill's "two spacers" rule),
 * and each bucket is direct-labeled with a swatch below the bar so
 * identity never rests on color alone.
 */
export function DistributionMeter({
  distribution,
}: {
  distribution: Distribution;
}) {
  const segments = [
    {
      key: "bullish",
      label: "Bullish",
      pct: distribution.bullish_pct,
      colorVar: "var(--bullish)",
    },
    {
      key: "neutral",
      label: "Neutral",
      pct: distribution.neutral_pct,
      colorVar: "var(--neutral)",
    },
    {
      key: "bearish",
      label: "Bearish",
      pct: distribution.bearish_pct,
      colorVar: "var(--bearish)",
    },
  ] as const;

  return (
    <div>
      <div
        className="flex h-6 w-full overflow-hidden rounded"
        style={{ backgroundColor: "var(--surface)" }}
        role="img"
        aria-label={`Bullish ${distribution.bullish_pct.toFixed(1)}%, neutral ${distribution.neutral_pct.toFixed(1)}%, bearish ${distribution.bearish_pct.toFixed(1)}%`}
      >
        {segments.map((segment, i) => (
          <div
            key={segment.key}
            className={
              i > 0
                ? "border-l-2"
                : undefined /* 2px surface-color gap between touching segments */
            }
            style={{
              width: `${segment.pct}%`,
              backgroundColor: segment.colorVar,
              borderColor: "var(--background)",
            }}
          />
        ))}
      </div>

      <dl className="mt-3 grid grid-cols-3 gap-3 text-sm">
        {segments.map((segment) => (
          <div key={segment.key} className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: segment.colorVar }}
              aria-hidden
            />
            <dt style={{ color: "var(--ink-secondary)" }}>{segment.label}</dt>
            <dd className="ml-auto font-semibold tabular-nums">
              {segment.pct.toFixed(1)}%
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
