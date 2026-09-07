import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
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
import {
  VoidStarProjection,
  type StudioAgentSummary,
  type StudioAssignmentSummary,
  type StudioHandoffSummary,
  type StudioSnapshot,
  type StudioUpdate,
  type VoidStarApi,
} from './voidstarProjection'

interface VoidStarFrame extends Window {
  VOIDSTAR?: VoidStarApi
  __vsGroundPoints?: () => number[][]
}

interface VoidStarWorldProps {
  agents: Agent[]
  onViewChange: (view: StudioView) => void
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
  onOpenAssignment: (assignmentId: string) => void
  onRouteHandoff: (sourceAssignmentId: string, toAgentId: string) => void
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

function snapshotFromRest(
  agents: Agent[],
  assignments: Assignment[],
  handoffs: Handoff[],
): StudioSnapshot {
  return {
    agents: agents as StudioAgentSummary[],
    assignments: assignments as StudioAssignmentSummary[],
    handoffs: handoffs as StudioHandoffSummary[],
  }
}

function eventLabel(update: StudioUpdate) {
  if (update.entity === 'agent') return `${update.item.name} · ${update.event}`
  if (update.entity === 'assignment') return `${update.item.title} · ${update.event}`
  return `${update.item.artifact_type} handoff · ${update.event}`
}

export function VoidStarWorld({
  agents,
  onViewChange,
  onManageAgent,
  onAssignAgent,
  onOpenAssignment,
  onRouteHandoff,
}: VoidStarWorldProps) {
  const qc = useQueryClient()
  const mainRef = useRef<HTMLElement>(null)
  const frameRef = useRef<HTMLIFrameElement>(null)
  const projectionRef = useRef<VoidStarProjection | null>(null)
  const pendingSnapshotRef = useRef<StudioSnapshot | null>(null)
  const pendingUpdatesRef = useRef<StudioUpdate[]>([])
  const [rendererReady, setRendererReady] = useState(false)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [latestEvent, setLatestEvent] = useState('Waiting for Studio state')
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
  const runningCount = assignments.filter(assignment => assignment.status === 'running').length
  const completedCount = assignments.filter(assignment => assignment.status === 'completed').length
  const restSnapshot = useMemo(
    () => snapshotFromRest(agents, assignments, handoffs),
    [agents, assignments, handoffs],
  )
  const saveLayout = useMutation({
    scope: { id: 'agent-world-layout' },
    mutationFn: api.updateAgentWorldLayout,
    onMutate: positions => {
      const previous = qc.getQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'])
      qc.setQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'], current => {
        const layout = current?.layout ?? EMPTY_LAYOUT
        return { layout: { ...layout, positions: { ...layout.positions, ...positions } } }
      })
      return { previous }
    },
    onSuccess: data => qc.setQueryData(['agent-world-layout'], data),
    onError: (_error, _positions, context) => {
      if (context?.previous) qc.setQueryData(['agent-world-layout'], context.previous)
      else void qc.invalidateQueries({ queryKey: ['agent-world-layout'] })
    },
  })
  const resetLayout = useMutation({
    scope: { id: 'agent-world-layout' },
    mutationFn: api.resetAgentWorldLayout,
    onMutate: () => {
      const previous = qc.getQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'])
      qc.setQueryData<{ layout: AgentWorldLayout }>(['agent-world-layout'], current => ({
        layout: { ...(current?.layout ?? EMPTY_LAYOUT), positions: {} },
      }))
      return { previous }
    },
    onSuccess: data => qc.setQueryData(['agent-world-layout'], data),
    onError: (_error, _variables, context) => {
      if (context?.previous) qc.setQueryData(['agent-world-layout'], context.previous)
      else void qc.invalidateQueries({ queryKey: ['agent-world-layout'] })
    },
  })
  const runAssignment = useMutation({
    mutationFn: api.runAssignment,
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['agents'] })
      void qc.invalidateQueries({ queryKey: ['assignments'] })
      void qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })

  useLayoutEffect(() => {
    const main = mainRef.current
    if (!main) return
    main.scrollTop = 0
    const frame = window.requestAnimationFrame(() => { main.scrollTop = 0 })
    return () => window.cancelAnimationFrame(frame)
  }, [])

  const renderSnapshot = useCallback((snapshot: StudioSnapshot) => {
    const projection = projectionRef.current
    if (!projection) {
      pendingSnapshotRef.current = snapshot
      return
    }
    projection.reset(snapshot)
    pendingSnapshotRef.current = null
    for (const update of pendingUpdatesRef.current) projection.apply(update)
    pendingUpdatesRef.current = []
    setLatestEvent('Studio state reconciled')
  }, [])

  const applyUpdate = useCallback((update: StudioUpdate) => {
    const projection = projectionRef.current
    if (projection) projection.apply(update)
    else pendingUpdatesRef.current.push(update)
    setLatestEvent(eventLabel(update))

    if (update.entity === 'agent') void qc.invalidateQueries({ queryKey: ['agents'] })
    if (update.entity === 'assignment') void qc.invalidateQueries({ queryKey: ['assignments'] })
    if (update.entity === 'handoff') void qc.invalidateQueries({ queryKey: ['handoffs'] })
  }, [qc])

  const handleFrameLoad = useCallback(() => {
    const frame = frameRef.current
    const world = frame?.contentWindow as unknown as VoidStarFrame | null
    const api = world?.VOIDSTAR
    const points = world?.__vsGroundPoints?.() ?? []
    if (!frame || !api || points.length === 0) {
      setLatestEvent('Renderer control surface unavailable')
      return
    }

    const document = frame.contentDocument
    if (document && !document.getElementById('agent-studio-overrides')) {
      const style = document.createElement('style')
      style.id = 'agent-studio-overrides'
      style.textContent = '#hud,#hint,#help,#palette{display:none!important}'
      document.head.appendChild(style)
    }

    projectionRef.current = new VoidStarProjection(api, points)
    setRendererReady(true)
    renderSnapshot(pendingSnapshotRef.current ?? restSnapshot)
  }, [renderSnapshot, restSnapshot])

  useEffect(() => {
    projectionRef.current?.updateAgentNames(agents as StudioAgentSummary[])
  }, [agents])

  useEffect(() => {
    if (!rendererReady || connection === 'live') return
    projectionRef.current?.reset(restSnapshot)
  }, [connection, rendererReady, restSnapshot])

  useEffect(() => {
    let disposed = false
    let socket: WebSocket | null = null
    let reconnectTimer: number | undefined

    function connect() {
      if (disposed) return
      setConnection('connecting')
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${location.host}/ws`)
      socket.addEventListener('open', () => setConnection('live'))
      socket.addEventListener('message', event => {
        try {
          const message = JSON.parse(String(event.data))
          if (message.type === 'init' && message.data?.agent_studio) {
            renderSnapshot(message.data.agent_studio as StudioSnapshot)
          }
          if (message.type === 'init' && message.data?.agent_world_layout) {
            qc.setQueryData(['agent-world-layout'], {
              layout: message.data.agent_world_layout as AgentWorldLayout,
            })
          }
          if (message.type === 'agent_studio_update') {
            applyUpdate(message.data as StudioUpdate)
          }
          if (message.type === 'agent_world_layout') {
            qc.setQueryData(['agent-world-layout'], {
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
  }, [applyUpdate, qc, renderSnapshot])

  return (
    <main ref={mainRef} className="min-h-0 flex flex-1 flex-col overflow-y-auto bg-neutral-950 lg:overflow-hidden">
      <header className="flex shrink-0 flex-col gap-5 border-b border-neutral-900 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
        <div>
          <div className="text-[10px] font-semibold tracking-[0.28em] uppercase text-cyan-400">Agent Studio · World</div>
          <h1 className="mt-2 text-2xl font-medium text-white">Work leaves a physical trace.</h1>
          <p className="mt-2 text-sm text-neutral-500">Lifecycle events only. No prompts, tokens, code, or private output.</p>
        </div>
        <StudioTabs view="world" onChange={onViewChange} />
      </header>

      <section className="relative min-h-[620px] flex-1 overflow-hidden bg-[#07060c] lg:min-h-0">
        <iframe
          ref={frameRef}
          title="Void Star live agent-work world"
          src="/voidstar/city.html?world=agent-studio"
          onLoad={handleFrameLoad}
          className="absolute inset-0 h-full w-full border-0"
        />

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

        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-4 bg-gradient-to-b from-neutral-950/90 via-neutral-950/45 to-transparent p-4 sm:p-6">
          <div className="border border-white/10 bg-neutral-950/70 px-3 py-2 backdrop-blur-md">
            <div className="flex items-center gap-2 text-[10px] font-semibold tracking-[0.2em] uppercase text-white/80">
              <i className={`h-1.5 w-1.5 rounded-full ${connection === 'live' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              {connection === 'live' ? 'Live state' : connection === 'retrying' ? 'Reconnecting' : 'Connecting'}
            </div>
            <div className="mt-1 max-w-[52vw] truncate text-xs text-white/50">{latestEvent}</div>
          </div>

          <div className="grid grid-cols-3 border border-white/10 bg-neutral-950/70 backdrop-blur-md">
            <WorldMetric label="Agents" value={agents.length} />
            <WorldMetric label="Running" value={runningCount} active />
            <WorldMetric label="Landed" value={completedCount} />
          </div>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-end justify-between gap-3 bg-gradient-to-t from-neutral-950/90 via-neutral-950/35 to-transparent p-4 pt-16 text-[10px] font-semibold tracking-[0.16em] uppercase text-white/45 sm:p-6 sm:pt-20">
          <div className="flex flex-wrap gap-x-5 gap-y-2">
            <span><b className="mr-1.5 text-amber-300">Beacon</b> active work</span>
            <span><b className="mr-1.5 text-cyan-300">Structure</b> artifact</span>
            <span><b className="mr-1.5 text-fuchsia-300">Signal</b> handoff</span>
            <span><b className="mr-1.5 text-pink-400">Port</b> drag to wire</span>
          </div>
          <span>{rendererReady ? 'Renderer online' : 'Renderer booting'}</span>
        </div>
      </section>
    </main>
  )
}

function WorldMetric({ label, value, active = false }: { label: string; value: number; active?: boolean }) {
  return (
    <div className="min-w-16 border-l border-white/10 px-3 py-2 text-right first:border-l-0 sm:min-w-20">
      <div className={`text-base tabular-nums ${active && value > 0 ? 'text-amber-300' : 'text-white'}`}>{value}</div>
      <div className="text-[9px] tracking-[0.16em] uppercase text-white/35">{label}</div>
    </div>
  )
}
