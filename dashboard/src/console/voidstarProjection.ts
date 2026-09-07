export interface StudioAgentSummary {
  id: string
  name: string
  status: 'idle' | 'running' | 'error'
  updated_at: string
}

export interface StudioAssignmentSummary {
  id: string
  title: string
  agent_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  source_handoff_id: string
  updated_at: string
  started_at: string | null
  completed_at: string | null
}

export interface StudioHandoffSummary {
  id: string
  source_assignment_id: string
  target_assignment_id: string
  from_agent_id: string
  to_agent_id: string
  artifact_type: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  updated_at: string
}

export interface StudioSnapshot {
  agents: StudioAgentSummary[]
  assignments: StudioAssignmentSummary[]
  handoffs: StudioHandoffSummary[]
}

export type StudioUpdate =
  | { entity: 'agent'; event: string; item: StudioAgentSummary }
  | { entity: 'assignment'; event: string; item: StudioAssignmentSummary }
  | { entity: 'handoff'; event: string; item: StudioHandoffSummary }

export interface VoidStarApi {
  startAgent: (cx: number, cz: number, color: number) => boolean
  completeAgent: (cx: number, cz: number, title: string) => boolean
  failAgent: (cx: number, cz: number, title: string) => boolean
  signal: (title: string, detail: string, tone?: string) => void
  setSpeed: (value: number) => void
  newWorld: () => void
}

interface WorkSite {
  agentId: string
  cx: number
  cz: number
}

const HISTORY_LIMIT = 24

function hash(value: string) {
  let result = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    result ^= value.charCodeAt(index)
    result = Math.imul(result, 16777619)
  }
  return result >>> 0
}

export class VoidStarProjection {
  private workSites = new Map<string, WorkSite>()
  private activeAssignmentByAgent = new Map<string, string>()
  private occupiedSites = new Set<number>()
  private agentNames = new Map<string, string>()
  private readonly world: VoidStarApi
  private readonly points: number[][]

  constructor(world: VoidStarApi, points: number[][]) {
    this.world = world
    this.points = points
  }

  reset(snapshot: StudioSnapshot) {
    this.workSites.clear()
    this.activeAssignmentByAgent.clear()
    this.occupiedSites.clear()
    this.agentNames = new Map(snapshot.agents.map(agent => [agent.id, agent.name]))
    this.world.newWorld()
    this.world.setSpeed(6)

    const assignments = [...snapshot.assignments]
      .sort((left, right) => left.updated_at.localeCompare(right.updated_at))
      .slice(-HISTORY_LIMIT)

    for (const assignment of assignments) this.applyAssignment(assignment.status, assignment)
    for (const agent of snapshot.agents) {
      if (agent.status === 'running' && !this.activeAssignmentByAgent.has(agent.id)) {
        this.startWork(`agent:${agent.id}`, agent.id)
      }
    }

    const latestHandoff = [...snapshot.handoffs]
      .sort((left, right) => left.updated_at.localeCompare(right.updated_at))
      .at(-1)
    if (latestHandoff) this.signalHandoff(latestHandoff)
  }

  updateAgentNames(agents: StudioAgentSummary[]) {
    this.agentNames = new Map(agents.map(agent => [agent.id, agent.name]))
  }

  apply(update: StudioUpdate) {
    if (update.entity === 'agent') this.applyAgent(update.event, update.item)
    if (update.entity === 'assignment') this.applyAssignment(update.event, update.item)
    if (update.entity === 'handoff') this.applyHandoff(update.event, update.item)
  }

  private applyAgent(event: string, agent: StudioAgentSummary) {
    this.agentNames.set(agent.id, agent.name)
    if (this.activeAssignmentByAgent.has(agent.id)) return
    const workId = `agent:${agent.id}`
    if (event === 'running') this.startWork(workId, agent.id)
    if (event === 'completed') this.completeWork(workId, `${agent.name} complete`)
    if (event === 'failed') this.failWork(workId, `${agent.name} blocked`)
  }

  private applyAssignment(event: string, assignment: StudioAssignmentSummary) {
    if (event === 'running') {
      this.activeAssignmentByAgent.set(assignment.agent_id, assignment.id)
      this.startWork(assignment.id, assignment.agent_id)
    }
    if (event === 'completed') {
      this.completeWork(assignment.id, `${this.agentName(assignment.agent_id)} complete`)
      this.activeAssignmentByAgent.delete(assignment.agent_id)
      this.world.signal('ARTIFACT LANDED', assignment.title, 'success')
    }
    if (event === 'failed') {
      this.failWork(assignment.id, `${this.agentName(assignment.agent_id)} blocked`)
      this.activeAssignmentByAgent.delete(assignment.agent_id)
    }
  }

  private applyHandoff(event: string, handoff: StudioHandoffSummary) {
    if (event === 'created') this.signalHandoff(handoff)
    if (event === 'running') {
      this.startWork(handoff.target_assignment_id, handoff.to_agent_id)
    }
    if (event === 'failed') {
      this.failWork(handoff.target_assignment_id, `${this.agentName(handoff.to_agent_id)} blocked`)
    }
  }

  private signalHandoff(handoff: StudioHandoffSummary) {
    const route = `${this.agentName(handoff.from_agent_id)} → ${this.agentName(handoff.to_agent_id)}`
    this.world.signal('HANDOFF ROUTED', `${handoff.artifact_type} · ${route}`, 'active')
  }

  private startWork(workId: string, agentId: string) {
    if (this.workSites.has(workId) || this.points.length === 0) return
    const start = hash(workId) % this.points.length
    for (let offset = 0; offset < this.points.length; offset += 1) {
      const index = (start + offset) % this.points.length
      if (this.occupiedSites.has(index)) continue
      const point = this.points[index]
      if (!point || !this.world.startAgent(point[2], point[3], hash(agentId) % 3)) continue
      this.occupiedSites.add(index)
      this.workSites.set(workId, { agentId, cx: point[2], cz: point[3] })
      return
    }
  }

  private completeWork(workId: string, title: string) {
    const site = this.workSites.get(workId)
    if (site) this.world.completeAgent(site.cx, site.cz, title)
  }

  private failWork(workId: string, title: string) {
    const site = this.workSites.get(workId)
    if (site) this.world.failAgent(site.cx, site.cz, title)
  }

  private agentName(agentId: string) {
    return this.agentNames.get(agentId) ?? 'Agent'
  }
}
