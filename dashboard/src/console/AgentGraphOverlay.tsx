import { useMemo, useRef, useState } from 'react'
import type { Agent, AgentWorldPoint, Assignment, Handoff } from '../lib/api'

interface AgentGraphOverlayProps {
  agents: Agent[]
  assignments: Assignment[]
  handoffs: Handoff[]
  positions: Record<string, AgentWorldPoint>
  layoutStatus: 'synced' | 'saving' | 'error'
  onSavePosition: (nodeId: string, point: AgentWorldPoint) => Promise<void>
  onResetLayout: () => Promise<void>
  runningAssignmentId: string | null
  assignmentRunTargetId: string | null
  assignmentRunError: string
  onRunAssignment: (assignmentId: string) => Promise<void>
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
  onOpenAssignment: (assignmentId: string) => void
  onRouteHandoff: (sourceAssignmentId: string, toAgentId: string) => void
}

type Point = AgentWorldPoint

interface GraphNode {
  id: string
  kind: 'agent' | 'assignment'
  agent?: Agent
  assignment?: Assignment
}

interface DragState {
  id: string
  startX: number
  startY: number
  origin: Point
  moved: boolean
}

interface WireDragState {
  sourceAssignmentId: string
  sourceNodeId: string
  sourceAgentId: string
  point: Point
  targetAgentId: string | null
  moved: boolean
}

const MIN_SEPARATION_X = 16
const MIN_SEPARATION_Y = 14

function clamp(value: number, minimum = 8, maximum = 92) {
  return Math.max(minimum, Math.min(maximum, value))
}

function positionsOverlap(left: Point, right: Point) {
  return Math.abs(left.x - right.x) < MIN_SEPARATION_X && Math.abs(left.y - right.y) < MIN_SEPARATION_Y
}

function laneGrid() {
  const slots: Point[] = []
  for (let y = 12; y <= 88; y += MIN_SEPARATION_Y) {
    for (let x = 8; x <= 92; x += MIN_SEPARATION_X) {
      slots.push({ x, y })
    }
  }
  return slots
}

function firstFreePoint(origin: Point, placed: Record<string, Point>): Point {
  const candidate = { x: clamp(origin.x), y: clamp(origin.y, 12, 88) }
  if (!Object.values(placed).some(other => positionsOverlap(candidate, other))) return candidate
  return [...laneGrid()]
    .sort((left, right) => {
      const leftDist = (left.x - candidate.x) ** 2 + (left.y - candidate.y) ** 2
      const rightDist = (right.x - candidate.x) ** 2 + (right.y - candidate.y) ** 2
      return leftDist - rightDist
    })
    .find(slot => !Object.values(placed).some(other => positionsOverlap(slot, other)))
    ?? candidate
}

function reconcileLayout(layout: Record<string, Point>, nodeIds: string[]) {
  const placed: Record<string, Point> = {}
  for (const nodeId of [...nodeIds].sort()) {
    const origin = layout[nodeId]
    if (!origin) continue
    placed[nodeId] = firstFreePoint(origin, placed)
  }
  return placed
}

function statusRank(status: Assignment['status']) {
  if (status === 'running') return 0
  if (status === 'queued') return 1
  if (status === 'failed') return 2
  return 3
}

function defaultLaneLayout(nodes: GraphNode[]) {
  const result: Record<string, Point> = {}
  const agents = nodes.filter(node => node.kind === 'agent')
  const columns = Math.min(4, Math.max(1, Math.ceil(Math.sqrt(agents.length || 1))))
  const rows = Math.ceil((agents.length || 1) / columns)

  agents.forEach((node, index) => {
    const column = index % columns
    const row = Math.floor(index / columns)
    const x = columns === 1 ? 50 : 16 + (column * 68) / (columns - 1)
    const y = rows === 1 ? 22 : 16 + (row * 36) / Math.max(rows - 1, 1)
    result[node.id] = { x, y }
    nodes
      .filter(item => item.assignment?.agent_id === node.agent?.id)
      .forEach((assignment, workIndex) => {
        result[assignment.id] = {
          x: clamp(x),
          y: clamp(y + 14 + workIndex * 14, 12, 88),
        }
      })
  })

  return result
}

function visibleWork(assignments: Assignment[], showCompleted: boolean) {
  const byAgent = new Map<string, Assignment[]>()
  for (const assignment of assignments) {
    if (!showCompleted && assignment.status === 'completed') continue
    const existing = byAgent.get(assignment.agent_id) ?? []
    existing.push(assignment)
    byAgent.set(assignment.agent_id, existing)
  }
  return [...byAgent.values()].flatMap(items => [...items]
    .sort((left, right) => {
      const rank = statusRank(left.status) - statusRank(right.status)
      return rank || right.updated_at.localeCompare(left.updated_at)
    })
    .slice(0, 3))
}

export function AgentGraphOverlay({
  agents,
  assignments,
  handoffs,
  positions,
  layoutStatus,
  onSavePosition,
  onResetLayout,
  runningAssignmentId,
  assignmentRunTargetId,
  assignmentRunError,
  onRunAssignment,
  onManageAgent,
  onAssignAgent,
  onOpenAssignment,
  onRouteHandoff,
}: AgentGraphOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<DragState | null>(null)
  const transientPositionsRef = useRef<Record<string, Point>>({})
  const wireDragRef = useRef<WireDragState | null>(null)
  const suppressWireClickRef = useRef(false)
  const [transientPositions, setTransientPositions] = useState<Record<string, Point>>({})
  const [wireDrag, setWireDrag] = useState<WireDragState | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [connectFrom, setConnectFrom] = useState<string | null>(null)
  const [showCompleted, setShowCompleted] = useState(false)
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const nodes = useMemo<GraphNode[]>(() => {
    const work = visibleWork(assignments, showCompleted)
    return agents.flatMap(agent => [
      { id: `agent:${agent.id}`, kind: 'agent' as const, agent },
      ...work
        .filter(assignment => assignment.agent_id === agent.id)
        .map(assignment => ({
          id: `assignment:${assignment.id}`,
          kind: 'assignment' as const,
          assignment,
        })),
    ])
  }, [agents, assignments, showCompleted])
  const defaults = useMemo(() => defaultLaneLayout(nodes), [nodes])
  const layout = useMemo(() => {
    const merged = { ...defaults, ...positions, ...transientPositions }
    if (draggingId) return merged
    return reconcileLayout(merged, nodes.map(node => node.id))
  }, [defaults, draggingId, nodes, positions, transientPositions])
  const selected = nodes.find(node => node.id === selectedId) ?? null
  const nodeIds = useMemo(() => new Set(nodes.map(node => node.id)), [nodes])
  const selectedSource = assignments.find(
    assignment => assignment.id === (connectFrom ?? wireDrag?.sourceAssignmentId),
  ) ?? null

  function selectNode(node: GraphNode) {
    if (connectFrom && node.agent) {
      if (node.agent.id !== selectedSource?.agent_id) {
        onRouteHandoff(connectFrom, node.agent.id)
        setConnectFrom(null)
      }
      return
    }
    setSelectedId(node.id)
  }

  function handlePointerDown(event: React.PointerEvent<HTMLButtonElement>, node: GraphNode) {
    if (!event.isPrimary || event.button !== 0) return
    const position = layout[node.id]
    if (!position) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = {
      id: node.id,
      startX: event.clientX,
      startY: event.clientY,
      origin: position,
      moved: false,
    }
    setDraggingId(node.id)
  }

  function handlePointerMove(event: React.PointerEvent<HTMLButtonElement>) {
    const drag = dragRef.current
    const bounds = containerRef.current?.getBoundingClientRect()
    if (!drag || !bounds) return
    const dx = ((event.clientX - drag.startX) / bounds.width) * 100
    const dy = ((event.clientY - drag.startY) / bounds.height) * 100
    if (Math.abs(dx) + Math.abs(dy) > 0.8) drag.moved = true
    setTransientPositions(current => {
      const next = {
        ...current,
        [drag.id]: { x: clamp(drag.origin.x + dx), y: clamp(drag.origin.y + dy, 12, 88) },
      }
      transientPositionsRef.current = next
      return next
    })
  }

  function handleNodeKeyDown(event: React.KeyboardEvent<HTMLButtonElement>, node: GraphNode) {
    const delta = event.shiftKey ? 5 : 2
    const current = layout[node.id]
    if (!current) return
    let next: Point | null = null
    if (event.key === 'ArrowLeft') next = { x: clamp(current.x - delta), y: current.y }
    if (event.key === 'ArrowRight') next = { x: clamp(current.x + delta), y: current.y }
    if (event.key === 'ArrowUp') next = { x: current.x, y: clamp(current.y - delta, 12, 88) }
    if (event.key === 'ArrowDown') next = { x: current.x, y: clamp(current.y + delta, 12, 88) }
    if (!next) return
    event.preventDefault()
    void onSavePosition(node.id, next).catch(() => undefined)
  }

  function handlePointerUp(node: GraphNode) {
    const drag = dragRef.current
    dragRef.current = null
    setDraggingId(null)
    if (!drag?.moved) selectNode(node)
    else {
      const point = transientPositionsRef.current[drag.id]
      if (!point) return
      const clearTransient = () => {
        setTransientPositions(current => {
          const currentPoint = current[drag.id]
          if (currentPoint?.x !== point.x || currentPoint?.y !== point.y) return current
          const next = { ...current }
          delete next[drag.id]
          transientPositionsRef.current = next
          return next
        })
      }
      void onSavePosition(drag.id, point).then(clearTransient, clearTransient)
    }
  }

  function handlePointerCancel() {
    const nodeId = dragRef.current?.id
    dragRef.current = null
    setDraggingId(null)
    if (!nodeId) return
    setTransientPositions(current => {
      if (!current[nodeId]) return current
      const next = { ...current }
      delete next[nodeId]
      transientPositionsRef.current = next
      return next
    })
  }

  function resetLayout() {
    transientPositionsRef.current = {}
    setTransientPositions({})
    void onResetLayout().catch(() => undefined)
  }

  function pointFromPointer(event: React.PointerEvent<HTMLButtonElement>) {
    const bounds = containerRef.current?.getBoundingClientRect()
    if (!bounds) return null
    return {
      x: clamp(((event.clientX - bounds.left) / bounds.width) * 100, 0, 100),
      y: clamp(((event.clientY - bounds.top) / bounds.height) * 100, 0, 100),
    }
  }

  function targetAgentAt(clientX: number, clientY: number, sourceAgentId: string) {
    const element = document.elementFromPoint(clientX, clientY)
    const target = element?.closest<HTMLElement>('[data-agent-target]')
    const agentId = target?.dataset.agentTarget ?? null
    return agentId && agentId !== sourceAgentId ? agentId : null
  }

  function handleWireStart(event: React.PointerEvent<HTMLButtonElement>, assignment: Assignment) {
    if (!event.isPrimary || event.button !== 0) return
    const sourceNodeId = `assignment:${assignment.id}`
    const point = layout[sourceNodeId]
    if (!point) return
    event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
    const next = {
      sourceAssignmentId: assignment.id,
      sourceNodeId,
      sourceAgentId: assignment.agent_id,
      point,
      targetAgentId: null,
      moved: false,
    }
    wireDragRef.current = next
    setWireDrag(next)
    setSelectedId(null)
  }

  function handleWireMove(event: React.PointerEvent<HTMLButtonElement>) {
    const drag = wireDragRef.current
    const point = pointFromPointer(event)
    if (!drag || !point) return
    const next = {
      ...drag,
      point,
      targetAgentId: targetAgentAt(event.clientX, event.clientY, drag.sourceAgentId),
      moved: drag.moved || Math.abs(point.x - layout[drag.sourceNodeId].x) + Math.abs(point.y - layout[drag.sourceNodeId].y) > 0.8,
    }
    wireDragRef.current = next
    setWireDrag(next)
  }

  function handleWireEnd() {
    const drag = wireDragRef.current
    suppressWireClickRef.current = Boolean(drag?.moved)
    wireDragRef.current = null
    setWireDrag(null)
    if (drag?.targetAgentId) {
      onRouteHandoff(drag.sourceAssignmentId, drag.targetAgentId)
    }
  }

  function handleWireCancel() {
    suppressWireClickRef.current = true
    wireDragRef.current = null
    setWireDrag(null)
  }

  function handleWireClick(event: React.MouseEvent<HTMLButtonElement>, assignmentId: string) {
    event.stopPropagation()
    if (suppressWireClickRef.current) {
      suppressWireClickRef.current = false
      return
    }
    setConnectFrom(assignmentId)
    setSelectedId(null)
  }

  return (
    <div ref={containerRef} className="pointer-events-none absolute inset-0 z-[5] overflow-hidden" aria-label="Agent topology graph">
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {nodes.filter(node => node.assignment).map(node => {
          const assignment = node.assignment!
          const from = layout[`agent:${assignment.agent_id}`]
          const to = layout[node.id]
          if (!from || !to) return null
          return <GraphEdge key={`hierarchy:${node.id}`} from={from} to={to} tone="hierarchy" />
        })}
        {handoffs.map(handoff => {
          const sourceAssignmentId = `assignment:${handoff.source_assignment_id}`
          const targetAssignmentId = `assignment:${handoff.target_assignment_id}`
          const sourceId = nodeIds.has(sourceAssignmentId) ? sourceAssignmentId : `agent:${handoff.from_agent_id}`
          const targetId = nodeIds.has(targetAssignmentId) ? targetAssignmentId : `agent:${handoff.to_agent_id}`
          const from = layout[sourceId]
          const to = layout[targetId]
          if (!from || !to) return null
          return <GraphEdge key={`handoff:${handoff.id}`} from={from} to={to} tone="handoff" />
        })}
        {wireDrag && layout[wireDrag.sourceNodeId] && (
          <GraphEdge from={layout[wireDrag.sourceNodeId]} to={wireDrag.point} tone="draft" />
        )}
      </svg>

      {nodes.map(node => {
        const position = layout[node.id]
        if (!position) return null
        const assignment = node.assignment
        const canWire = assignment?.status === 'completed' && Boolean(assignment.result)
        const isWireTarget = Boolean(node.agent && wireDrag?.targetAgentId === node.agent.id)
        const isTargetMode = Boolean(node.agent && (connectFrom || wireDrag))
        return (
          <div
            key={node.id}
            data-graph-node={node.id}
            className={`pointer-events-none absolute touch-none select-none ${node.kind === 'agent' ? 'w-44 lg:w-48' : 'w-36 lg:w-40'}`}
            style={{ left: `${position.x}%`, top: `${position.y}%`, transform: 'translate(-50%, -50%)', zIndex: selectedId === node.id ? 6 : node.kind === 'agent' ? 3 : 2 }}
          >
            <button
              data-agent-target={node.agent?.id}
              type="button"
              aria-label={node.agent
                ? `${node.agent.name}. ${node.agent.mission || 'No mission defined'}. Status: ${node.agent.status}`
                : `${assignment?.title} assignment node`}
              onPointerDown={event => handlePointerDown(event, node)}
              onPointerMove={handlePointerMove}
              onPointerUp={() => handlePointerUp(node)}
              onPointerCancel={handlePointerCancel}
              onKeyDown={event => handleNodeKeyDown(event, node)}
              onClick={event => { if (event.detail === 0) selectNode(node) }}
              className={`pointer-events-auto w-full border text-left transition-[border-color,background-color,box-shadow] ${node.kind === 'agent' ? 'px-2 py-2 sm:px-3 sm:py-2.5' : 'px-2 py-2 sm:px-2.5'} ${selectedId === node.id ? 'border-cyan-300 bg-[#0b1217] shadow-[0_0_24px_rgba(34,211,238,0.16)]' : isWireTarget ? 'border-fuchsia-200 bg-fuchsia-950 shadow-[0_0_24px_rgba(232,121,249,0.24)]' : isTargetMode ? 'border-fuchsia-400/70 bg-fuchsia-950 hover:border-fuchsia-200' : 'border-white/15 bg-[#0b0e12] hover:border-white/40'}`}
            >
              {node.agent ? <AgentNode agent={node.agent} /> : <AssignmentNode assignment={assignment!} />}
            </button>
            {canWire && (
              <button
                type="button"
                aria-label={`Wire ${assignment.title} to another agent`}
                title="Drag to a target agent"
                onPointerDown={event => handleWireStart(event, assignment)}
                onPointerMove={handleWireMove}
                onPointerUp={handleWireEnd}
                onPointerCancel={handleWireCancel}
                onClick={event => handleWireClick(event, assignment.id)}
                className="pointer-events-auto absolute left-1/2 top-full z-10 h-4 w-4 -translate-x-1/2 -translate-y-1/2 touch-none rounded-full border border-fuchsia-200 bg-fuchsia-500 shadow-[0_0_16px_rgba(232,121,249,0.8)] transition-transform hover:scale-125 focus:scale-125 focus:outline-none sm:-right-2 sm:left-auto sm:top-1/2 sm:translate-x-0"
              />
            )}
          </div>
        )
      })}

      <div className="pointer-events-auto absolute bottom-16 right-4 flex items-center border border-white/10 bg-[#080a0d]/95 text-[9px] font-semibold tracking-[0.14em] uppercase sm:right-6">
        <span role="status" aria-live="polite" className={`border-r border-white/10 px-2.5 py-1.5 ${layoutStatus === 'error' ? 'text-red-300' : layoutStatus === 'saving' ? 'text-amber-300' : 'text-emerald-300/60'}`}>
          {layoutStatus === 'error' ? 'Save failed' : layoutStatus === 'saving' ? 'Saving' : 'Layout synced'}
        </span>
        <button
          type="button"
          aria-pressed={showCompleted}
          aria-label={showCompleted ? 'Hide completed assignments' : 'Show completed assignments'}
          onClick={() => setShowCompleted(current => !current)}
          className={`border-r border-white/10 px-2.5 py-1.5 ${showCompleted ? 'text-cyan-200' : 'text-white/45 hover:text-white'}`}
        >
          {showCompleted ? 'Completed shown' : 'Completed hidden'}
        </button>
        <button type="button" aria-label="Reset layout" disabled={layoutStatus === 'saving'} onClick={resetLayout} className="px-2.5 py-1.5 text-white/45 hover:text-white disabled:opacity-30">
          Reset
        </button>
      </div>

      {(connectFrom || wireDrag) && (
        <div className="pointer-events-auto absolute left-1/2 top-20 -translate-x-1/2 border border-fuchsia-400/50 bg-fuchsia-950 px-4 py-2 text-center text-xs text-fuchsia-100">
          {wireDrag ? 'Drop on a target agent' : `Select a target agent for “${selectedSource?.title}”`}
          {connectFrom && <button type="button" onClick={() => setConnectFrom(null)} className="ml-3 text-fuchsia-300/60 hover:text-white">Cancel</button>}
        </div>
      )}

      {selected && !connectFrom && !wireDrag && (
        <NodeInspector
          node={selected}
          onClose={() => setSelectedId(null)}
          onManageAgent={onManageAgent}
          onAssignAgent={onAssignAgent}
          onOpenAssignment={onOpenAssignment}
          onConnect={setConnectFrom}
          runningAssignmentId={runningAssignmentId}
          assignmentRunTargetId={assignmentRunTargetId}
          runError={assignmentRunError}
          onRunAssignment={onRunAssignment}
        />
      )}
    </div>
  )
}

function GraphEdge({ from, to, tone }: { from: Point; to: Point; tone: 'hierarchy' | 'handoff' | 'draft' }) {
  const bend = Math.max(4, Math.abs(to.y - from.y) * 0.45)
  const path = `M ${from.x} ${from.y} C ${from.x} ${from.y + bend}, ${to.x} ${to.y - bend}, ${to.x} ${to.y}`
  return (
    <path
      d={path}
      vectorEffect="non-scaling-stroke"
      fill="none"
      stroke={tone === 'draft' ? 'rgba(244,114,182,0.95)' : tone === 'handoff' ? 'rgba(232,121,249,0.72)' : 'rgba(103,232,249,0.24)'}
      strokeWidth={tone === 'hierarchy' ? 1 : 1.5}
      strokeDasharray={tone === 'hierarchy' ? undefined : '5 5'}
    />
  )
}

function AgentNode({ agent }: { agent: Agent }) {
  return (
    <>
      <div className="flex items-start justify-between gap-2">
        <span title={agent.name} className="line-clamp-2 min-w-0 text-xs font-semibold leading-snug text-white">{agent.name}</span>
        <i aria-hidden="true" className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${agent.status === 'running' ? 'bg-amber-300' : agent.status === 'error' ? 'bg-red-400' : 'bg-emerald-400'}`} />
      </div>
      <div className="mt-2 text-[8px] font-semibold tracking-[0.14em] uppercase text-cyan-300/55">Mission</div>
      <div title={agent.mission} className="mt-0.5 line-clamp-2 text-[10px] leading-relaxed text-white/55">{agent.mission || 'No mission defined'}</div>
      <div className={`mt-2 text-[8px] font-semibold tracking-[0.14em] uppercase ${agent.status === 'running' ? 'text-amber-300' : agent.status === 'error' ? 'text-red-300' : 'text-emerald-300/60'}`}>{agent.status}</div>
    </>
  )
}

function AssignmentNode({ assignment }: { assignment: Assignment }) {
  return (
    <>
      <div className="line-clamp-2 text-[10px] font-medium leading-snug text-white/85">{assignment.title}</div>
      <div className={`mt-1.5 text-[8px] tracking-[0.14em] uppercase ${assignment.status === 'running' ? 'text-amber-300' : assignment.status === 'failed' ? 'text-red-300' : assignment.status === 'completed' ? 'text-cyan-300' : 'text-white/35'}`}>{assignment.status}</div>
    </>
  )
}

interface NodeInspectorProps {
  node: GraphNode
  onClose: () => void
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
  onOpenAssignment: (assignmentId: string) => void
  onConnect: (assignmentId: string) => void
  runningAssignmentId: string | null
  assignmentRunTargetId: string | null
  runError: string
  onRunAssignment: (assignmentId: string) => Promise<void>
}

function NodeInspector({ node, onClose, onManageAgent, onAssignAgent, onOpenAssignment, onConnect, runningAssignmentId, assignmentRunTargetId, runError, onRunAssignment }: NodeInspectorProps) {
  const agent = node.agent
  const assignment = node.assignment
  const runnable = assignment?.status === 'queued' || assignment?.status === 'failed'
  const isRunning = assignment?.id === runningAssignmentId
  return (
    <aside className="pointer-events-auto absolute right-4 top-20 w-[min(280px,calc(100%-2rem))] border border-white/15 bg-[#090c10] p-4 text-left shadow-2xl sm:right-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[9px] font-semibold tracking-[0.18em] uppercase text-cyan-300/70">{node.kind} controls</div>
          <div className="mt-1 text-sm font-medium text-white">{agent?.name ?? assignment?.title}</div>
        </div>
        <button type="button" onClick={onClose} className="text-lg leading-none text-white/30 hover:text-white" aria-label="Close node controls">×</button>
      </div>
      <p className="mt-3 line-clamp-4 text-xs leading-relaxed text-white/45">{agent?.mission ?? assignment?.objective}</p>
      {assignment && <div className="mt-3 text-[9px] font-semibold tracking-[0.16em] uppercase text-white/35">{assignment.status} · {assignment.priority} priority</div>}
      {assignment?.id === assignmentRunTargetId && runError && <div className="mt-3 border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">{runError}</div>}
      <div className="mt-4 flex flex-wrap gap-2">
        {agent && <button type="button" onClick={() => onAssignAgent(agent.id)} className="bg-cyan-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950">Assign work</button>}
        {agent && <button type="button" onClick={() => onManageAgent(agent.id)} className="border border-white/15 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/70 hover:border-white/40">Edit agent</button>}
        {runnable && <button type="button" disabled={isRunning} onClick={() => { void onRunAssignment(assignment.id).catch(() => undefined) }} className="bg-amber-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950 disabled:cursor-wait disabled:opacity-50">{isRunning ? 'Mini working' : assignment.status === 'failed' ? 'Retry assignment' : 'Run assignment'}</button>}
        {assignment && <button type="button" onClick={() => onOpenAssignment(assignment.id)} className="bg-cyan-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950">Open work</button>}
        {assignment?.status === 'completed' && assignment.result && <button type="button" onClick={() => onConnect(assignment.id)} className="border border-fuchsia-400/50 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-fuchsia-200 hover:border-fuchsia-200">Connect handoff</button>}
      </div>
    </aside>
  )
}
