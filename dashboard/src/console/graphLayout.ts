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

export interface SuiteMember { suite?: string }

export interface SuiteGroup {
  slug: string
  count: number
  color: string
}

export function suiteOf(agent: SuiteMember) {
  return (agent.suite ?? '').trim().toLowerCase()
}

// Hash the slug so a suite keeps its colour across sessions, filters and new suites.
// Hues 270-329 are skipped so no suite reads as a magenta handoff edge.
export function suiteColor(slug: string) {
  let hash = 0x811c9dc5
  for (const char of slug) {
    hash ^= char.codePointAt(0) ?? 0
    hash = Math.imul(hash, 0x01000193)
  }
  const hue = (hash >>> 0) % 300
  return `hsl(${hue < 270 ? hue : hue + 60} 85% 70%)`
}

export function suiteGroups(agents: SuiteMember[]): SuiteGroup[] {
  const counts = new Map<string, number>()
  for (const agent of agents) {
    const slug = suiteOf(agent)
    if (slug) counts.set(slug, (counts.get(slug) ?? 0) + 1)
  }
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([slug, count]) => ({ slug, count, color: suiteColor(slug) }))
}

// Suite members sit next to each other in the default layout; agents without a
// suite follow. Order inside a group is the incoming order.
export function orderBySuite<T extends SuiteMember>(agents: T[]) {
  return agents
    .map((agent, index) => ({ agent, index, slug: suiteOf(agent) }))
    .sort((left, right) => {
      if (Boolean(left.slug) !== Boolean(right.slug)) return left.slug ? -1 : 1
      return left.slug.localeCompare(right.slug) || left.index - right.index
    })
    .map(item => item.agent)
}

export function filterBySuite<T extends SuiteMember>(agents: T[], slug: string | null) {
  return slug ? agents.filter(agent => suiteOf(agent) === slug) : agents
}
