import assert from 'node:assert/strict'
import test from 'node:test'
import { GRAPH_WIDTH, graphGeometry, reconcileLayout } from '../src/console/graphLayout.ts'

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
