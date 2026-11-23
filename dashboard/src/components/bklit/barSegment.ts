// From Bklit UI, packages/ui/src/charts/sparkline-geometry.ts (MIT).
const round = (value: number) => Math.round(value * 100) / 100;

/** One bar as a path segment, rounding the corners away from the baseline. */
export function buildSparklineBarSegment(
  cx: number,
  barWidth: number,
  y: number,
  baselineY: number,
  radius: number
): string {
  const height = Math.abs(baselineY - y);
  if (height === 0 || barWidth <= 0) {
    return "";
  }
  const x0 = round(cx - barWidth / 2);
  const x1 = round(cx + barWidth / 2);
  const r = Math.min(radius, barWidth / 2, height);
  const up = y < baselineY;
  const tip = round(y);
  const base = round(baselineY);
  if (r <= 0) {
    return `M${x0},${base}V${tip}H${x1}V${base}Z`;
  }
  const dir = up ? 1 : -1;
  const rr = round(r);
  const sweep = up ? 1 : 0;
  return (
    `M${x0},${base}V${round(tip + dir * r)}` +
    `A${rr},${rr} 0 0 ${sweep} ${round(x0 + r)},${tip}` +
    `H${round(x1 - r)}` +
    `A${rr},${rr} 0 0 ${sweep} ${x1},${round(tip + dir * r)}` +
    `V${base}Z`
  );
}
