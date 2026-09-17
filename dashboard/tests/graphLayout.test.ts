import assert from 'node:assert/strict'
import test from 'node:test'
import {
  GRAPH_WIDTH,
  filterBySuite,
  graphGeometry,
  orderBySuite,
  reconcileLayout,
  suiteColor,
  suiteGroups,
  suiteOf,
} from '../src/console/graphLayout.ts'

for (const count of [0, 1, 24, 36, 48, 96, 160]) {
  test(`lays out ${count} colliding nodes without overlap or clipping`, () => {
    const ids = Array.from({ length: count }, (_, i) => `agent:${i}`)
    const saved = Object.fromEntries(ids.map(id => [id, { x: 8, y: 12 }]))
    const result = reconcileLayout(saved, ids)
    assert.equal(Object.keys(result).length, count)
    assert.deepEqual(result, reconcileLayout(saved, [...ids].reverse()))
    const { height } = graphGeometry(count)
    const points = Object.values(result)
    for (const point of points) {
      assert.ok(point.x / 100 * GRAPH_WIDTH >= 96)
      assert.ok((100 - point.x) / 100 * GRAPH_WIDTH >= 96)
      assert.ok(point.y / 100 * height >= 72)
      assert.ok((100 - point.y) / 100 * height >= 72)
    }
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const dx = Math.abs(points[i].x - points[j].x) / 100 * GRAPH_WIDTH
        const dy = Math.abs(points[i].y - points[j].y) / 100 * height
        assert.ok(dx >= 192 || dy >= 144, `Overlap: ${i}, ${j}`)
      }
    }
  })
}

test('preserves valid positions without mutating stored layout', () => {
  const saved = { a: { x: 20, y: 20 }, b: { x: 60, y: 60 } }
  assert.deepEqual(reconcileLayout(saved, ['a', 'b']), saved)
})

test('handles missing and invalid positions', () => {
  const result = reconcileLayout({ a: { x: NaN, y: Infinity }, b: { x: -5, y: 150 } }, ['a', 'b', 'c'])
  for (const point of Object.values(result)) {
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y))
    assert.ok(point.x >= 8 && point.x <= 92 && point.y >= 12 && point.y <= 88)
  }
})

const agent = (id: string, suite?: string) => ({ id, suite })

test('suite slugs are normalized and blank means no suite', () => {
  assert.equal(suiteOf(agent('a', ' Research ')), 'research')
  assert.equal(suiteOf(agent('a', '')), '')
  assert.equal(suiteOf(agent('a')), '')
})

test('a suite always gets the same valid colour, and distinct suites differ', () => {
  assert.equal(suiteColor('mindmoor'), suiteColor('mindmoor'))
  assert.match(suiteColor('mindmoor'), /^hsl\(\d{1,3} 85% 70%\)$/)
  const colours = new Set(['research', 'review', 'mindmoor', 'ops'].map(suiteColor))
  assert.equal(colours.size, 4)
  for (let i = 0; i < 500; i++) {
    const hue = Number(suiteColor(`suite-${i}`).match(/^hsl\((\d+) /)?.[1])
    assert.ok(hue >= 0 && hue < 360 && (hue < 270 || hue >= 330), `suite-${i} hue ${hue}`)
  }
})

test('groups count members per suite, sorted, ignoring agents without a suite', () => {
  const agents = [agent('a', 'review'), agent('b', 'research'), agent('c'), agent('d', 'Research')]
  assert.deepEqual(suiteGroups(agents), [
    { slug: 'research', count: 2, color: suiteColor('research') },
    { slug: 'review', count: 1, color: suiteColor('review') },
  ])
  assert.deepEqual(suiteGroups([agent('a'), agent('b', '')]), [])
  assert.deepEqual(suiteGroups([]), [])
})

test('no suites keeps the incoming order untouched', () => {
  const agents = [agent('c'), agent('a'), agent('b')]
  assert.deepEqual(orderBySuite(agents), agents)
})

test('suite members are placed together, stable within a suite, unsuited last', () => {
  const agents = [agent('x'), agent('r2', 'review'), agent('s1', 'research'), agent('y'), agent('r1', 'review'), agent('s2', 'research')]
  assert.deepEqual(orderBySuite(agents).map(item => item.id), ['s1', 's2', 'r2', 'r1', 'x', 'y'])
  assert.deepEqual(agents.map(item => item.id), ['x', 'r2', 's1', 'y', 'r1', 's2'])
})

test('filtering by a suite keeps only its members and clearing restores all', () => {
  const agents = [agent('a', 'review'), agent('b', 'research'), agent('c')]
  assert.deepEqual(filterBySuite(agents, 'review').map(item => item.id), ['a'])
  assert.deepEqual(filterBySuite(agents, 'missing'), [])
  assert.equal(filterBySuite(agents, null), agents)
})

test('many suites still lay out without overlap', () => {
  const agents = Array.from({ length: 90 }, (_, i) => agent(`agent-${i}`, i % 3 ? `suite-${i % 45}` : ''))
  const groups = suiteGroups(agents)
  assert.equal(groups.length, 30)
  assert.equal(groups.reduce((total, group) => total + group.count, 0), 60)
  const ids = orderBySuite(agents).map(item => `agent:${item.id}`)
  assert.equal(new Set(ids).size, 90)
  const result = reconcileLayout(Object.fromEntries(ids.map(id => [id, { x: 50, y: 50 }])), ids)
  const { height } = graphGeometry(ids.length)
  const points = Object.values(result)
  assert.equal(points.length, 90)
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = Math.abs(points[i].x - points[j].x) / 100 * GRAPH_WIDTH
      const dy = Math.abs(points[i].y - points[j].y) / 100 * height
      assert.ok(dx >= 192 || dy >= 144, `Overlap: ${i}, ${j}`)
    }
  }
})
