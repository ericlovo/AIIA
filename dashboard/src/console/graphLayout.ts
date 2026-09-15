export interface Point { x: number; y: number }

export const GRAPH_WIDTH = 1440

export function graphGeometry(count: number) {
  const rows = Math.max(6, Math.ceil(count / 6))
  return { rows, stepY: 76 / (rows - 1), height: Math.ceil(rows * 180 / 0.76) }
}

export function reconcileLayout(layout: Record<string, Point>, nodeIds: string[]) {
  const ids = [...nodeIds].sort()
  const { rows, stepY } = graphGeometry(ids.length)
  const slots: Point[] = Array.from({ length: rows * 6 }, (_, i) => ({
    x: 8 + (i % 6) * 16.8,
    y: 12 + Math.floor(i / 6) * stepY,
  }))
  const placed: Record<string, Point> = {}
  const overlaps = (point: Point) => Object.values(placed).some(other =>
    Math.abs(point.x - other.x) < 16 && Math.abs(point.y - other.y) < stepY - 0.01,
  )
  for (const id of ids) {
    const saved = layout[id]
    const origin = saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)
      ? { x: Math.max(8, Math.min(92, saved.x)), y: Math.max(12, Math.min(88, saved.y)) }
      : slots[ids.indexOf(id)]
    const point = !overlaps(origin) ? origin : [...slots]
      .sort((a, b) => ((a.x - origin.x) ** 2 + (a.y - origin.y) ** 2) - ((b.x - origin.x) ** 2 + (b.y - origin.y) ** 2))
      .find(candidate => !overlaps(candidate))
    // Arbitrary saved positions can consume multiple grid slots. Reflow instead of stacking.
    if (!point) return Object.fromEntries(ids.map((nodeId, i) => [nodeId, slots[i]]))
    placed[id] = point
  }
  return placed
}
