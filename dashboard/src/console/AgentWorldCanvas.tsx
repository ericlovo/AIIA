import { useEffect, useState } from 'react'
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

interface AgentWorldCanvasProps {
  agents: Agent[]
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

function graphMinHeight(agentCount: number, assignments: Assignment[]) {
  const assignmentCounts = new Map<string, number>()
  for (const assignment of assignments) {
    assignmentCounts.set(assignment.agent_id, (assignmentCounts.get(assignment.agent_id) ?? 0) + 1)
  }
  const visibleAssignments = [...assignmentCounts.values()].reduce(
    (total, count) => total + Math.min(count, 3),
    0,
  )
  const nodeCount = agentCount + visibleAssignments
  const columns = Math.min(5, Math.max(1, Math.ceil(Math.sqrt(nodeCount))))
  const rows = Math.ceil(nodeCount / columns)
  return Math.max(620, rows * 128 + 180)
}

export function AgentWorldCanvas({
  agents,
  onViewChange,
  onManageAgent,
  onAssignAgent,
  onOpenAssignment,
  onRouteHandoff,
}: AgentWorldCanvasProps) {
  const queryClient = useQueryClient()
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [latestEvent, setLatestEvent] = useState('Loading agent graph')
  const { data: assignmentData } = useQuery({
    queryKey: ['assignments'],
    queryFn: api.assignments,
    refetchInterval: connection === 'live' ? false : 5_000,
  })
  const { data: handoffData } = useQuery({
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
  const mapMinHeight = graphMinHeight(agents.length, assignments)
  const activeCount = agents.filter(agent => agent.status === 'running').length
    + assignments.filter(assignment => assignment.status === 'running').length

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
    <main className="min-h-0 flex flex-1 flex-col overflow-y-auto bg-neutral-950 lg:overflow-hidden">
      <header className="flex shrink-0 flex-col gap-4 border-b border-neutral-900 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-7">
        <div>
          <div className="text-[10px] font-semibold tracking-[0.28em] uppercase text-cyan-400">Agent Studio · Map</div>
          <h1 className="mt-1.5 text-xl font-medium text-white">Agent control map</h1>
          <p className="mt-1 text-xs text-neutral-500">Drag nodes to organize work. Select a node to act.</p>
        </div>
        <StudioTabs view="world" onChange={onViewChange} />
      </header>

      <section className="relative min-h-[620px] max-w-full flex-1 overflow-auto bg-[#080a0d] lg:min-h-0">
        <div className="relative h-full min-w-[1000px] lg:min-w-0" style={{ minHeight: mapMinHeight }}>
          <SpatialGrid />
          <AgentGraphOverlay
          agents={agents}
          assignments={assignments}
          handoffs={handoffs}
          positions={layoutState.positions}
          layoutStatus={layoutLoadError || saveLayout.isError || resetLayout.isError ? 'error' : saveLayout.isPending || resetLayout.isPending ? 'saving' : 'synced'}
          onSavePosition={(nodeId: string, point: AgentWorldPoint) => saveLayout.mutateAsync({ [nodeId]: point }).then(() => undefined)}
          onResetLayout={() => resetLayout.mutateAsync().then(() => undefined)}
          runningAssignmentId={runAssignment.isPending ? runAssignment.variables : null}
          assignmentRunTargetId={runAssignment.variables ?? null}
          assignmentRunError={runAssignment.error?.message ?? ''}
          onRunAssignment={assignmentId => runAssignment.mutateAsync(assignmentId).then(() => undefined)}
          onManageAgent={onManageAgent}
          onAssignAgent={onAssignAgent}
          onOpenAssignment={onOpenAssignment}
          onRouteHandoff={onRouteHandoff}
          />

          <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-4 p-4 sm:p-6">
            <div className="border border-white/10 bg-[#080a0d]/90 px-3 py-2">
              <div className="flex items-center gap-2 text-[9px] font-semibold tracking-[0.18em] uppercase text-white/70">
                <i className={`h-1.5 w-1.5 rounded-full ${connection === 'live' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                {connection === 'live' ? 'Live' : connection === 'retrying' ? 'Reconnecting' : 'Connecting'}
              </div>
              <div className="mt-1 max-w-[48vw] truncate text-[11px] text-white/35">{latestEvent}</div>
            </div>

            <div className="grid grid-cols-3 border border-white/10 bg-[#080a0d]/90">
              <MapMetric label="Agents" value={agents.length} />
              <MapMetric label="Active" value={activeCount} active />
              <MapMetric label="Work" value={assignments.length} />
            </div>
          </div>

          <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-center justify-between gap-3 border-t border-white/8 bg-[#080a0d]/90 px-4 py-3 text-[9px] font-semibold tracking-[0.14em] uppercase text-white/35 sm:px-6">
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <span><b className="mr-1.5 text-cyan-300">Solid</b>assignment</span>
              <span><b className="mr-1.5 text-fuchsia-300">Dashed</b>handoff</span>
              <span><b className="mr-1.5 text-white/70">Drag</b>to position</span>
            </div>
            <span>Click a node for controls</span>
          </div>
        </div>
      </section>
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

function MapMetric({ label, value, active = false }: { label: string; value: number; active?: boolean }) {
  return (
    <div className="min-w-14 border-l border-white/10 px-2.5 py-2 text-right first:border-l-0 sm:min-w-20 sm:px-3">
      <div className={`text-sm tabular-nums ${active && value > 0 ? 'text-amber-300' : 'text-white'}`}>{value}</div>
      <div className="text-[8px] tracking-[0.14em] uppercase text-white/30">{label}</div>
    </div>
  )
}
