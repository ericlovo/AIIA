import { useEffect, useMemo, useRef, useState } from 'react'
import { Maximize2, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type Agent,
  type AgentWorldLayout,
  type AgentWorldPoint,
  type Assignment,
  type Handoff,
} from '../lib/api'
import { AgentGraphOverlay } from './AgentGraphOverlay'
import { StudioTabs, type StudioView } from './StudioTabs'
import { GRAPH_WIDTH, filterBySuite, graphGeometry, suiteGroups } from './graphLayout'
import { SuiteLegend } from './SuiteLegend'
import { withCreatedAssignment, withCreatedHandoff, withoutHandoff } from './mapRelationships'

interface AgentWorldCanvasProps {
  agents: Agent[]
  loading: boolean
  agentError: boolean
  onViewChange: (view: StudioView) => void
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
  onOpenAssignment: (assignmentId: string) => void
  onRouteHandoff: (sourceAssignmentId: string, toAgentId: string) => void
}

interface StudioUpdate {
  entity: 'agent' | 'assignment' | 'handoff'
  event: string
  item: { name?: string; title?: string; artifact_type?: string }
}

type ConnectionState = 'connecting' | 'live' | 'retrying'

const EMPTY_ASSIGNMENTS: Assignment[] = []
const EMPTY_HANDOFFS: Handoff[] = []
const EMPTY_LAYOUT: AgentWorldLayout = {
  version: 1,
  revision: 0,
  positions: {},
  updated_at: null,
}

function eventLabel(update: StudioUpdate) {
  const label = update.item.name ?? update.item.title ?? `${update.item.artifact_type ?? 'work'} handoff`
  return `${label} · ${update.event}`
}

function graphMinHeight(agentCount: number, assignments: Assignment[], showCompleted: boolean) {
  const assignmentCounts = new Map<string, number>()
  for (const assignment of assignments) {
    if (!showCompleted && assignment.status === 'completed') continue
    assignmentCounts.set(assignment.agent_id, (assignmentCounts.get(assignment.agent_id) ?? 0) + 1)
  }
  const visibleAssignments = [...assignmentCounts.values()].reduce(
    (total, count) => total + Math.min(count, 3),
    0,
  )
  const nodeCount = agentCount + visibleAssignments
  return graphGeometry(nodeCount).height
}

export function AgentWorldCanvas({
  agents,
  loading,
  agentError,
  onViewChange,
  onManageAgent,
  onAssignAgent,
  onOpenAssignment,
  onRouteHandoff,
}: AgentWorldCanvasProps) {
  const queryClient = useQueryClient()
  const [showCompleted, setShowCompleted] = useState(false)
  const [suiteFilter, setSuiteFilter] = useState<string | null>(null)
  const [inspectorHost, setInspectorHost] = useState<HTMLDivElement | null>(null)
  const viewportRef = useRef<HTMLElement>(null)
  const [zoom, setZoom] = useState(1)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [latestEvent, setLatestEvent] = useState('Loading agent graph')
  const { data: assignmentData, isError: assignmentLoadError, isPending: assignmentLoading } = useQuery({
    queryKey: ['assignments'],
    queryFn: api.assignments,
    refetchInterval: connection === 'live' ? false : 5_000,
  })
  const { data: handoffData, isError: handoffLoadError } = useQuery({
    queryKey: ['handoffs'],
    queryFn: api.handoffs,
    refetchInterval: connection === 'live' ? false : 5_000,
  })
  const { data: layoutData, isError: layoutLoadError } = useQuery({
    queryKey: ['agent-world-layout'],
    queryFn: api.agentWorldLayout,
    refetchInterval: connection === 'live' ? false : 5_000,
  })
  const assignments = assignmentData?.assignments ?? EMPTY_ASSIGNMENTS
  const handoffs = handoffData?.handoffs ?? EMPTY_HANDOFFS
  const layoutState = layoutData?.layout ?? EMPTY_LAYOUT
  const suites = useMemo(() => suiteGroups(agents), [agents])
  const activeSuite = suites.some(group => group.slug === suiteFilter) ? suiteFilter : null
  const mapAgents = useMemo(() => filterBySuite(agents, activeSuite), [agents, activeSuite])
  const mapAssignments = useMemo(() => {
    if (!activeSuite) return assignments
    const members = new Set(mapAgents.map(agent => agent.id))
    return assignments.filter(assignment => members.has(assignment.agent_id))
  }, [activeSuite, assignments, mapAgents])
  const mapMinHeight = graphMinHeight(mapAgents.length, mapAssignments, showCompleted)
  const activeCount = agents.filter(agent => agent.status === 'running').length

  const saveLayout = useMutation({
    scope: { id: 'agent-world-layout' },
    mutationFn: api.updateAgentWorldLayout,
    onMutate: positions => {
      const previous = queryClient.getQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'])
      queryClient.setQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'], current => {
        const layout = current?.layout ?? EMPTY_LAYOUT
        return { layout: { ...layout, positions: { ...layout.positions, ...positions } } }
      })
      return { previous }
    },
    onSuccess: data => queryClient.setQueryData(['agent-world-layout'], data),
    onError: (_error, _positions, context) => {
      if (context?.previous) queryClient.setQueryData(['agent-world-layout'], context.previous)
      else void queryClient.invalidateQueries({ queryKey: ['agent-world-layout'] })
    },
  })
  const resetLayout = useMutation({
    scope: { id: 'agent-world-layout' },
    mutationFn: api.resetAgentWorldLayout,
    onMutate: () => {
      const previous = queryClient.getQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'])
      queryClient.setQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'], current => ({
        layout: { ...(current?.layout ?? EMPTY_LAYOUT), positions: {} },
      }))
      return { previous }
    },
    onSuccess: data => queryClient.setQueryData(['agent-world-layout'], data),
    onError: (_error, _variables, context) => {
      if (context?.previous) queryClient.setQueryData(['agent-world-layout'], context.previous)
      else void queryClient.invalidateQueries({ queryKey: ['agent-world-layout'] })
    },
  })
  const createHandoff = useMutation({
    mutationFn: api.createHandoff,
    onSuccess: ({ handoff, assignment }) => {
      queryClient.setQueryData<{ handoffs: Handoff[] }>(['handoffs'], current => ({
        handoffs: withCreatedHandoff(current?.handoffs ?? EMPTY_HANDOFFS, handoff),
      }))
      queryClient.setQueryData<{ assignments: Assignment[] }>(['assignments'], current => ({
        assignments: withCreatedAssignment(current?.assignments ?? EMPTY_ASSIGNMENTS, assignment),
      }))
      void queryClient.invalidateQueries({ queryKey: ['assignments'] })
      void queryClient.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const deleteHandoff = useMutation({
    mutationFn: api.deleteHandoff,
    onSuccess: (_result, handoffId) => {
      queryClient.setQueryData<{ handoffs: Handoff[] }>(['handoffs'], current => ({
        handoffs: withoutHandoff(current?.handoffs ?? EMPTY_HANDOFFS, handoffId),
      }))
      void queryClient.invalidateQueries({ queryKey: ['assignments'] })
      void queryClient.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const runAssignment = useMutation({
    mutationFn: api.runAssignment,
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['agents'] })
      void queryClient.invalidateQueries({ queryKey: ['assignments'] })
      void queryClient.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })

  useEffect(() => {
    let disposed = false
    let socket: WebSocket | null = null
    let reconnectTimer: number | undefined

    function invalidateStudio(entity?: StudioUpdate['entity']) {
      if (!entity || entity === 'agent') void queryClient.invalidateQueries({ queryKey: ['agents'] })
      if (!entity || entity === 'assignment') void queryClient.invalidateQueries({ queryKey: ['assignments'] })
      if (!entity || entity === 'handoff') void queryClient.invalidateQueries({ queryKey: ['handoffs'] })
    }

    function connect() {
      if (disposed) return
      setConnection('connecting')
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${location.host}/ws`)
      socket.addEventListener('open', () => setConnection('live'))
      socket.addEventListener('message', event => {
        try {
          const message = JSON.parse(String(event.data))
          if (message.type === 'init') {
            invalidateStudio()
            if (message.data?.agent_world_layout) {
              queryClient.setQueryData(['agent-world-layout'], {
                layout: message.data.agent_world_layout as AgentWorldLayout,
              })
            }
            setLatestEvent('Agent graph reconciled')
          }
          if (message.type === 'agent_studio_update') {
            const update = message.data as StudioUpdate
            invalidateStudio(update.entity)
            setLatestEvent(eventLabel(update))
          }
          if (message.type === 'agent_world_layout') {
            queryClient.setQueryData(['agent-world-layout'], {
              layout: message.data as AgentWorldLayout,
            })
          }
        } catch {
          setLatestEvent('Ignored malformed Studio event')
        }
      })
      socket.addEventListener('close', () => {
        if (disposed) return
        setConnection('retrying')
        reconnectTimer = window.setTimeout(connect, 1_500)
      })
      socket.addEventListener('error', () => setConnection('retrying'))
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [queryClient])

  return (
    <main className="h-full min-h-0 flex flex-1 flex-col overflow-hidden bg-neutral-950 [&_*]:tracking-normal">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-4 border-b border-neutral-900 px-5 py-4 sm:px-7">
        <div className="min-w-48">
          <div className="text-[10px] font-semibold uppercase text-cyan-400">Agent Studio / Map</div>
          <h1 className="mt-1.5 text-xl font-medium text-white">Agent control map</h1>
        </div>
        <StudioTabs view="world" onChange={onViewChange} />
      </header>
      <div className="flex shrink-0 flex-wrap items-center gap-4 border-b border-white/10 px-5 py-3 text-xs text-neutral-400">
        <span>{loading ? 'Loading agents' : `${activeSuite ? `${mapAgents.length} of ${agents.length}` : agents.length} agents / ${activeCount} running`} / {assignmentLoading ? 'Loading work' : `${assignments.length} assignments`}</span>
        <label className="flex items-center gap-2"><input type="checkbox" checked={showCompleted} onChange={event => setShowCompleted(event.target.checked)} />Completed assignments</label>
        <button type="button" title="Reset layout" aria-label="Reset layout" disabled={saveLayout.isPending || resetLayout.isPending} onClick={() => { saveLayout.reset(); resetLayout.mutate() }} className="flex h-8 w-8 items-center justify-center border border-neutral-700 disabled:opacity-40"><RotateCcw size={15} /></button>
        <span role="status">{layoutLoadError ? 'Layout unavailable' : saveLayout.isError || resetLayout.isError ? 'Layout save failed' : saveLayout.isPending || resetLayout.isPending ? 'Saving layout' : layoutData ? 'Layout synced' : 'Loading layout'}</span>
        <span title={latestEvent}>{connection === 'live' ? 'Live' : connection === 'retrying' ? 'Reconnecting' : 'Connecting'}</span>
        <div className="flex items-center gap-2" role="group" aria-label="Map zoom">
          <button type="button" aria-label="Zoom out" title="Zoom out" disabled={zoom <= 0.15} className="flex h-8 w-8 items-center justify-center border border-neutral-700 disabled:opacity-40" onClick={() => setZoom(value => Math.max(0.15, value - 0.15))}><ZoomOut size={15} /></button>
          <output className="w-10 text-center">{Math.round(zoom * 100)}%</output>
          <button type="button" aria-label="Zoom in" title="Zoom in" disabled={zoom >= 1.5} className="flex h-8 w-8 items-center justify-center border border-neutral-700 disabled:opacity-40" onClick={() => setZoom(value => Math.min(1.5, value + 0.15))}><ZoomIn size={15} /></button>
          <button type="button" aria-label="Fit map" title="Fit map" className="flex h-8 w-8 items-center justify-center border border-neutral-700" onClick={() => {
            const viewport = viewportRef.current
            if (!viewport) return
            setZoom(Math.min(1, viewport.clientWidth / GRAPH_WIDTH, viewport.clientHeight / mapMinHeight))
            viewport.scrollTo(0, 0)
          }}><Maximize2 size={15} /></button>
        </div>
        <SuiteLegend groups={suites} activeSuite={activeSuite} onSelect={setSuiteFilter} />
      </div>
      {(agentError || assignmentLoadError || handoffLoadError || layoutLoadError) && <div role="alert" className="px-5 py-2 text-xs text-amber-200">Map data is incomplete. <button type="button" className="underline" onClick={() => { for (const key of ['agents', 'assignments', 'handoffs', 'agent-world-layout']) void queryClient.invalidateQueries({ queryKey: [key] }) }}>Retry</button></div>}
      {!loading && !agentError && agents.length === 0 && <p role="status" className="px-5 py-3 text-sm text-neutral-400">No agents configured.</p>}
      <div className="relative min-h-0 flex-1">
      <section ref={viewportRef} aria-label="Scrollable agent map" tabIndex={0} className="h-full max-w-full overflow-auto bg-[#080a0d]">
        <div className="relative" style={{ width: GRAPH_WIDTH * zoom, height: mapMinHeight * zoom }}>
        <div className="absolute left-0 top-0 origin-top-left" style={{ width: GRAPH_WIDTH, height: mapMinHeight, transform: `scale(${zoom})` }}>
          <SpatialGrid />
          <AgentGraphOverlay
          inspectorHost={inspectorHost}
          showCompleted={showCompleted}
          agents={mapAgents}
          assignments={mapAssignments}
          handoffs={handoffs}
          positions={layoutState.positions}
          onSavePosition={(nodeId: string, point: AgentWorldPoint) => { resetLayout.reset(); return saveLayout.mutateAsync({ [nodeId]: point }).then(() => undefined) }}
          runningAssignmentId={runAssignment.isPending ? runAssignment.variables : null}
          assignmentRunTargetId={runAssignment.variables ?? null}
          assignmentRunError={runAssignment.error?.message ?? ''}
          onRunAssignment={assignmentId => runAssignment.mutateAsync(assignmentId).then(() => undefined)}
          onManageAgent={onManageAgent}
          onAssignAgent={onAssignAgent}
          onOpenAssignment={onOpenAssignment}
          onRouteHandoff={onRouteHandoff}
          onCreateHandoff={data => createHandoff.mutateAsync(data).then(result => result.handoff)}
          onDeleteHandoff={handoffId => deleteHandoff.mutateAsync(handoffId).then(() => undefined)}
          />

          <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-center justify-between gap-3 border-t border-white/8 bg-[#080a0d]/90 px-4 py-3 text-[9px] font-semibold tracking-[0.14em] uppercase text-white/35 sm:px-6">
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <span><b className="mr-1.5 text-cyan-300">Solid</b>assignment</span>
              <span><b className="mr-1.5 text-fuchsia-300">Dashed</b>handoff · click to inspect</span>
              <span><b className="mr-1.5 text-white/70">Drag</b>to position</span>
            </div>
            <span>Click a node for controls</span>
          </div>
        </div>
        </div>
      </section>
      <div ref={setInspectorHost} className="pointer-events-none absolute inset-0 z-20" />
      </div>
    </main>
  )
}

function SpatialGrid() {
  return (
    <div
      aria-hidden="true"
      className="absolute inset-0"
      style={{
        backgroundImage: [
          'linear-gradient(rgba(34,211,238,0.055) 1px, transparent 1px)',
          'linear-gradient(90deg, rgba(34,211,238,0.055) 1px, transparent 1px)',
          'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px)',
          'linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
        ].join(','),
        backgroundSize: '160px 160px, 160px 160px, 32px 32px, 32px 32px',
        backgroundPosition: 'center center',
      }}
    >
      <div className="absolute inset-y-0 left-1/2 border-l border-cyan-300/10" />
      <div className="absolute inset-x-0 top-1/2 border-t border-cyan-300/10" />
    </div>
  )
}
