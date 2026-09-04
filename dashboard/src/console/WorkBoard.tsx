import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type Agent,
  type Assignment,
  type AssignmentDefinition,
  type AssignmentPriority,
  type AssignmentStatus,
  type Handoff,
  type HandoffArtifactType,
  type HandoffDefinition,
  type GitWorkspace,
  type RepositoryResource,
} from '../lib/api'
import { StudioTabs, type StudioView } from './StudioTabs'

const EMPTY_ASSIGNMENTS: Assignment[] = []
const EMPTY_HANDOFFS: Handoff[] = []
const EMPTY_WORKSPACES: GitWorkspace[] = []
const EMPTY_REPOS: RepositoryResource[] = []
const PRIORITIES: AssignmentPriority[] = ['low', 'normal', 'high', 'urgent']
const ARTIFACT_TYPES: HandoffArtifactType[] = ['brief', 'analysis', 'plan', 'decision', 'review']

const EMPTY_ASSIGNMENT: AssignmentDefinition = {
  title: '',
  objective: '',
  agent_id: '',
  priority: 'normal',
  context: '',
  success_criteria: '',
}

const EMPTY_HANDOFF: HandoffDefinition = {
  source_assignment_id: '',
  to_agent_id: '',
  artifact_type: 'brief',
  instructions: '',
}

interface WorkBoardProps {
  agents: Agent[]
  view: Exclude<StudioView, 'agents'>
  onViewChange: (view: StudioView) => void
}

export function WorkBoard({ agents, view, onViewChange }: WorkBoardProps) {
  const qc = useQueryClient()
  const { data: assignmentData, isLoading: assignmentsLoading } = useQuery({
    queryKey: ['assignments'],
    queryFn: api.assignments,
    refetchInterval: 5_000,
  })
  const { data: handoffData, isLoading: handoffsLoading } = useQuery({
    queryKey: ['handoffs'],
    queryFn: api.handoffs,
    refetchInterval: 5_000,
  })
  const { data: workspaceData } = useQuery({
    queryKey: ['git-workspaces'],
    queryFn: api.gitWorkspaces,
    refetchInterval: 5_000,
  })
  const { data: resourceData } = useQuery({
    queryKey: ['agent-resources'],
    queryFn: api.agentResources,
  })
  const assignments = assignmentData?.assignments ?? EMPTY_ASSIGNMENTS
  const handoffs = handoffData?.handoffs ?? EMPTY_HANDOFFS
  const workspaces = workspaceData?.workspaces ?? EMPTY_WORKSPACES
  const repos = resourceData?.repos ?? EMPTY_REPOS
  const [selectedAssignmentId, setSelectedAssignmentId] = useState<string | null>(null)
  const [selectedHandoffId, setSelectedHandoffId] = useState<string | null>(null)
  const [assignmentDraft, setAssignmentDraft] = useState(EMPTY_ASSIGNMENT)
  const [handoffDraft, setHandoffDraft] = useState(EMPTY_HANDOFF)
  const selectedAssignment = assignments.find(item => item.id === selectedAssignmentId) ?? null
  const selectedHandoff = handoffs.find(item => item.id === selectedHandoffId) ?? null

  const createAssignment = useMutation({
    mutationFn: api.createAssignment,
    onSuccess: ({ assignment }) => {
      setSelectedAssignmentId(assignment.id)
      setAssignmentDraft(EMPTY_ASSIGNMENT)
      qc.invalidateQueries({ queryKey: ['assignments'] })
    },
  })
  const runAssignment = useMutation({
    mutationFn: api.runAssignment,
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['agents'] })
      qc.invalidateQueries({ queryKey: ['assignments'] })
      qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const removeAssignment = useMutation({
    mutationFn: api.deleteAssignment,
    onSuccess: () => {
      setSelectedAssignmentId(null)
      qc.invalidateQueries({ queryKey: ['assignments'] })
      qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const createHandoff = useMutation({
    mutationFn: api.createHandoff,
    onSuccess: ({ handoff }) => {
      setSelectedHandoffId(handoff.id)
      setHandoffDraft(EMPTY_HANDOFF)
      qc.invalidateQueries({ queryKey: ['assignments'] })
      qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const removeHandoff = useMutation({
    mutationFn: api.deleteHandoff,
    onSuccess: () => {
      setSelectedHandoffId(null)
      qc.invalidateQueries({ queryKey: ['assignments'] })
      qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  const requestWorkspace = useMutation({
    mutationFn: api.requestGitWorkspace,
    onSettled: () => qc.invalidateQueries({ queryKey: ['git-workspaces'] }),
  })
  const approveWorkspace = useMutation({
    mutationFn: api.approveGitWorkspace,
    onSettled: () => qc.invalidateQueries({ queryKey: ['git-workspaces'] }),
  })

  const completedAssignments = assignments.filter(item => item.status === 'completed' && item.result)
  const runningCount = assignments.filter(item => item.status === 'running').length

  function agentName(agentId: string) {
    return agents.find(agent => agent.id === agentId)?.name ?? 'Removed agent'
  }

  function prepareHandoff(assignment: Assignment) {
    setHandoffDraft({
      source_assignment_id: assignment.id,
      to_agent_id: '',
      artifact_type: 'brief',
      instructions: '',
    })
    setSelectedHandoffId(null)
    onViewChange('handoffs')
  }

  const loading = view === 'assignments' ? assignmentsLoading : handoffsLoading
  const items = view === 'assignments' ? assignments : handoffs

  return (
    <main className="min-h-0 flex-1 grid grid-cols-1 overflow-y-auto bg-neutral-950 lg:grid-cols-[minmax(0,1fr)_390px] lg:overflow-hidden">
      <section className="min-w-0 border-b border-neutral-900 lg:overflow-hidden lg:border-r lg:border-b-0">
        <div className="flex flex-col gap-5 border-b border-neutral-900 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
          <div>
            <div className="text-[10px] font-semibold tracking-[0.28em] uppercase text-cyan-400">Agent Studio</div>
            <h1 className="mt-2 text-2xl font-medium text-white">{view === 'assignments' ? 'Assignment queue' : 'Handoff ledger'}</h1>
            <div className="mt-2 flex gap-4 text-xs text-neutral-500">
              <span>{assignments.length} assignments</span>
              <span>{runningCount} running</span>
              <span>{handoffs.length} handoffs</span>
            </div>
          </div>
          <StudioTabs view={view} onChange={onViewChange} />
        </div>

        <div className="min-h-[540px] overflow-y-auto px-5 py-6 sm:px-7 lg:h-[calc(100%-118px)]">
          <div className="mx-auto max-w-5xl">
            <div className="mb-4 flex items-center justify-between gap-4">
              <div className="text-[10px] font-semibold tracking-[0.2em] uppercase text-neutral-600">
                {view === 'assignments' ? 'Work ledger' : 'Artifact routes'}
              </div>
              <button
                onClick={() => view === 'assignments' ? setSelectedAssignmentId(null) : setSelectedHandoffId(null)}
                className="border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:border-cyan-500/60 hover:text-cyan-200"
              >
                {view === 'assignments' ? 'New assignment' : 'New handoff'}
              </button>
            </div>

            {!loading && items.length === 0 && (
              <button
                onClick={() => view === 'assignments' ? setSelectedAssignmentId(null) : setSelectedHandoffId(null)}
                className="min-h-48 w-full border border-dashed border-neutral-700 p-6 text-left hover:border-cyan-500/50"
              >
                <div className="text-base text-white">{view === 'assignments' ? 'No assignments yet' : 'No handoffs yet'}</div>
                <div className="mt-2 text-xs text-neutral-500">{view === 'assignments' ? 'Create focused work for one local agent.' : 'Route a completed artifact to the next specialist.'}</div>
              </button>
            )}

            {view === 'assignments' ? (
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                {assignments.map(assignment => (
                  <AssignmentCard
                    key={assignment.id}
                    assignment={assignment}
                    agentName={agentName(assignment.agent_id)}
                    selected={selectedAssignmentId === assignment.id}
                    onSelect={() => {
                      runAssignment.reset()
                      removeAssignment.reset()
                      setSelectedAssignmentId(assignment.id)
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {handoffs.map(handoff => (
                  <HandoffCard
                    key={handoff.id}
                    handoff={handoff}
                    fromAgent={agentName(handoff.from_agent_id)}
                    toAgent={agentName(handoff.to_agent_id)}
                    selected={selectedHandoffId === handoff.id}
                    onSelect={() => {
                      runAssignment.reset()
                      removeHandoff.reset()
                      setSelectedHandoffId(handoff.id)
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <aside className="min-h-0 overflow-y-auto bg-neutral-950">
        {view === 'assignments' ? (
          selectedAssignment ? (
            <AssignmentDetails
              assignment={selectedAssignment}
              agent={agents.find(item => item.id === selectedAssignment.agent_id) ?? null}
              agentName={agentName(selectedAssignment.agent_id)}
              workspace={workspaces.find(item => item.assignment_id === selectedAssignment.id) ?? null}
              repo={repos.find(item => item.id === agents.find(agent => agent.id === selectedAssignment.agent_id)?.repo_id) ?? null}
              isRunning={runAssignment.isPending}
              isRemoving={removeAssignment.isPending}
              isRequestingWorkspace={requestWorkspace.isPending}
              isApprovingWorkspace={approveWorkspace.isPending}
              hasHandoff={handoffs.some(item => item.source_assignment_id === selectedAssignment.id || item.target_assignment_id === selectedAssignment.id)}
              error={runAssignment.error ?? removeAssignment.error ?? requestWorkspace.error ?? approveWorkspace.error}
              onRun={() => runAssignment.mutate(selectedAssignment.id)}
              onHandoff={() => prepareHandoff(selectedAssignment)}
              onRequestWorkspace={() => requestWorkspace.mutate(selectedAssignment.id)}
              onApproveWorkspace={workspaceId => approveWorkspace.mutate(workspaceId)}
              onRemove={() => removeAssignment.mutate(selectedAssignment.id)}
            />
          ) : (
            <AssignmentForm
              agents={agents}
              draft={assignmentDraft}
              pending={createAssignment.isPending}
              error={createAssignment.error}
              onChange={setAssignmentDraft}
              onSubmit={() => createAssignment.mutate(assignmentDraft)}
            />
          )
        ) : selectedHandoff ? (
          <HandoffDetails
            handoff={selectedHandoff}
            assignment={assignments.find(item => item.id === selectedHandoff.target_assignment_id) ?? null}
            fromAgent={agentName(selectedHandoff.from_agent_id)}
            toAgent={agentName(selectedHandoff.to_agent_id)}
            isRunning={runAssignment.isPending}
            isRemoving={removeHandoff.isPending}
            error={runAssignment.error ?? removeHandoff.error}
            onRun={() => runAssignment.mutate(selectedHandoff.target_assignment_id)}
            onRemove={() => removeHandoff.mutate(selectedHandoff.id)}
          />
        ) : (
          <HandoffForm
            agents={agents}
            assignments={completedAssignments}
            draft={handoffDraft}
            pending={createHandoff.isPending}
            error={createHandoff.error}
            onChange={setHandoffDraft}
            onSubmit={() => createHandoff.mutate(handoffDraft)}
          />
        )}
      </aside>
    </main>
  )
}

function AssignmentCard({ assignment, agentName, selected, onSelect }: { assignment: Assignment; agentName: string; selected: boolean; onSelect: () => void }) {
  return (
    <button onClick={onSelect} className={`min-h-40 border p-5 text-left transition-colors ${selected ? 'border-cyan-400/70 bg-cyan-500/10' : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-600'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-base font-medium text-white">{assignment.title}</div>
          <div className="mt-1 text-xs text-cyan-300/80">{agentName}</div>
        </div>
        <StatusLabel status={assignment.status} />
      </div>
      <p className="mt-4 line-clamp-2 text-xs leading-relaxed text-neutral-500">{assignment.objective}</p>
      <div className="mt-5 flex items-center justify-between text-[10px] uppercase tracking-[0.16em] text-neutral-600">
        <span>{assignment.priority} priority</span>
        <span>{relativeTime(assignment.updated_at)}</span>
      </div>
    </button>
  )
}

function HandoffCard({ handoff, fromAgent, toAgent, selected, onSelect }: { handoff: Handoff; fromAgent: string; toAgent: string; selected: boolean; onSelect: () => void }) {
  return (
    <button onClick={onSelect} className={`w-full border p-5 text-left transition-colors ${selected ? 'border-cyan-400/70 bg-cyan-500/10' : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-600'}`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-center gap-3 text-sm">
          <span className="truncate text-white">{fromAgent}</span>
          <span aria-hidden="true" className="text-cyan-400">→</span>
          <span className="truncate text-white">{toAgent}</span>
        </div>
        <StatusLabel status={handoff.status} />
      </div>
      <div className="mt-3 text-[10px] uppercase tracking-[0.18em] text-cyan-300/70">{handoff.artifact_type}</div>
      <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-neutral-500">{handoff.instructions}</p>
    </button>
  )
}

function AssignmentForm({ agents, draft, pending, error, onChange, onSubmit }: { agents: Agent[]; draft: AssignmentDefinition; pending: boolean; error: Error | null; onChange: (draft: AssignmentDefinition) => void; onSubmit: () => void }) {
  return (
    <Panel title="New assignment" eyebrow="Assignment controls">
      <Field label="Title"><input value={draft.title} onChange={event => onChange({ ...draft, title: event.target.value })} placeholder="Map the authorization surface" /></Field>
      <Field label="Assigned agent">
        <select value={draft.agent_id} onChange={event => onChange({ ...draft, agent_id: event.target.value })}>
          <option value="">Choose an agent</option>
          {agents.map(agent => <option key={agent.id} value={agent.id}>{agent.name}</option>)}
        </select>
      </Field>
      <Field label="Priority">
        <select value={draft.priority} onChange={event => onChange({ ...draft, priority: event.target.value as AssignmentPriority })}>
          {PRIORITIES.map(priority => <option key={priority} value={priority}>{priority}</option>)}
        </select>
      </Field>
      <Field label="Objective"><textarea value={draft.objective} onChange={event => onChange({ ...draft, objective: event.target.value })} rows={5} placeholder="Return the five highest-leverage integration points." /></Field>
      <Field label="Context"><textarea value={draft.context} onChange={event => onChange({ ...draft, context: event.target.value })} rows={4} placeholder="Relevant decisions, constraints, or source material." /></Field>
      <Field label="Success criteria"><textarea value={draft.success_criteria} onChange={event => onChange({ ...draft, success_criteria: event.target.value })} rows={3} placeholder="What must be true for this work to be done?" /></Field>
      {error && <ErrorNotice error={error} />}
      <button disabled={!draft.title.trim() || !draft.objective.trim() || !draft.agent_id || pending} onClick={onSubmit} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{pending ? 'Creating…' : 'Create assignment'}</button>
    </Panel>
  )
}

interface AssignmentDetailsProps {
  assignment: Assignment
  agent: Agent | null
  agentName: string
  workspace: GitWorkspace | null
  repo: RepositoryResource | null
  isRunning: boolean
  isRemoving: boolean
  isRequestingWorkspace: boolean
  isApprovingWorkspace: boolean
  hasHandoff: boolean
  error: Error | null
  onRun: () => void
  onHandoff: () => void
  onRequestWorkspace: () => void
  onApproveWorkspace: (workspaceId: string) => void
  onRemove: () => void
}

function AssignmentDetails({ assignment, agent, agentName, workspace, repo, isRunning, isRemoving, isRequestingWorkspace, isApprovingWorkspace, hasHandoff, error, onRun, onHandoff, onRequestWorkspace, onApproveWorkspace, onRemove }: AssignmentDetailsProps) {
  const runnable = assignment.status === 'queued' || assignment.status === 'failed'
  const gitEnabled = agent?.tools.includes('Git workspace') ?? false
  return (
    <Panel title={assignment.title} eyebrow="Assignment controls">
      <Meta label="Owner" value={agentName} />
      <div className="grid grid-cols-2 gap-3"><Meta label="Status" value={assignment.status} /><Meta label="Priority" value={assignment.priority} /></div>
      <TextBlock label="Objective" value={assignment.objective} />
      {assignment.success_criteria && <TextBlock label="Success criteria" value={assignment.success_criteria} />}
      {assignment.context && <TextBlock label="Context" value={assignment.context} muted />}
      {assignment.result && <TextBlock label="Work product" value={assignment.result} />}
      {assignment.error && <div className="border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">{assignment.error}</div>}
      {assignment.status === 'completed' && (
        <GitWorkspacePanel
          agentName={agentName}
          gitEnabled={gitEnabled}
          repo={repo}
          workspace={workspace}
          isRequesting={isRequestingWorkspace}
          isApproving={isApprovingWorkspace}
          onRequest={onRequestWorkspace}
          onApprove={onApproveWorkspace}
        />
      )}
      {error && <ErrorNotice error={error} />}
      {runnable && <button disabled={isRunning} onClick={onRun} className="w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{isRunning ? 'Mini is working…' : assignment.status === 'failed' ? 'Retry assignment' : 'Run assignment'}</button>}
      {assignment.status === 'completed' && <button onClick={onHandoff} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950">Hand off work</button>}
      <button disabled={isRemoving || assignment.status === 'running' || hasHandoff} onClick={onRemove} title={hasHandoff ? 'Unlink the handoff before deleting connected work' : undefined} className="w-full px-3 py-2 text-xs text-neutral-600 hover:text-red-300 disabled:opacity-30">Delete assignment</button>
    </Panel>
  )
}

function GitWorkspacePanel({ agentName, gitEnabled, repo, workspace, isRequesting, isApproving, onRequest, onApprove }: { agentName: string; gitEnabled: boolean; repo: RepositoryResource | null; workspace: GitWorkspace | null; isRequesting: boolean; isApproving: boolean; onRequest: () => void; onApprove: (workspaceId: string) => void }) {
  const eligible = repo?.git_workspace?.eligible ?? false
  const blocked = repo && repo.git_workspace?.eligible === false
  return (
    <div className="border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold tracking-[0.16em] uppercase text-cyan-400">Git workspace</div>
          <p className="mt-1 text-xs text-neutral-500">Isolated branch on the Mini. No commit or push access.</p>
        </div>
        {workspace && <WorkspaceStatus status={workspace.status} />}
      </div>

      {!gitEnabled && <p className="mt-4 text-xs leading-relaxed text-amber-300/80">Enable Git workspace on {agentName} before requesting implementation space.</p>}
      {gitEnabled && !repo && <p className="mt-4 text-xs text-amber-300/80">This agent needs a mounted repository.</p>}
      {gitEnabled && blocked && <p className="mt-4 text-xs leading-relaxed text-amber-300/80">Workspace blocked: this mount shares a GitHub remote with another repository. Correct the remote first.</p>}

      {gitEnabled && eligible && !workspace && (
        <button disabled={isRequesting} onClick={onRequest} className="mt-4 w-full border border-cyan-500/50 px-3 py-2.5 text-sm text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-40">{isRequesting ? 'Requesting…' : 'Request Git workspace'}</button>
      )}

      {workspace?.status === 'pending' && (
        <div className="mt-4 space-y-3">
          <WorkspaceMeta label="Proposed branch" value={workspace.branch} />
          <p className="text-xs leading-relaxed text-neutral-500">Approval creates the branch and worktree. It does not run the agent or change files.</p>
          <button disabled={isApproving} onClick={() => onApprove(workspace.id)} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:opacity-40">{isApproving ? 'Preparing…' : 'Approve & create worktree'}</button>
        </div>
      )}

      {workspace?.status === 'preparing' && <p className="mt-4 text-xs text-amber-300">Preparing isolated worktree…</p>}
      {workspace?.status === 'failed' && <p className="mt-4 break-words text-xs leading-relaxed text-red-300">{workspace.error || 'Workspace creation failed.'}</p>}
      {workspace?.status === 'ready' && (
        <div className="mt-4 space-y-3">
          <WorkspaceMeta label="Branch" value={workspace.branch} />
          <WorkspaceMeta label="Base" value={workspace.base_ref} />
          <WorkspaceMeta label="Mini path" value={workspace.path} />
          <pre className="max-h-32 overflow-auto whitespace-pre-wrap border border-neutral-800 bg-neutral-950 p-3 text-[11px] leading-relaxed text-neutral-400">{workspace.git_status}</pre>
        </div>
      )}
    </div>
  )
}

function WorkspaceMeta({ label, value }: { label: string; value: string }) {
  return <div><div className="text-[10px] uppercase tracking-[0.14em] text-neutral-600">{label}</div><div className="mt-1 break-all font-mono text-[11px] text-neutral-300">{value}</div></div>
}

function WorkspaceStatus({ status }: { status: GitWorkspace['status'] }) {
  const color = status === 'ready' ? 'border-green-800 text-green-300' : status === 'failed' ? 'border-red-900 text-red-300' : 'border-amber-700 text-amber-300'
  return <span className={`shrink-0 border px-2 py-1 text-[10px] uppercase tracking-[0.12em] ${color}`}>{status}</span>
}

function HandoffForm({ agents, assignments, draft, pending, error, onChange, onSubmit }: { agents: Agent[]; assignments: Assignment[]; draft: HandoffDefinition; pending: boolean; error: Error | null; onChange: (draft: HandoffDefinition) => void; onSubmit: () => void }) {
  const source = assignments.find(item => item.id === draft.source_assignment_id)
  const targets = agents.filter(agent => agent.id !== source?.agent_id)
  return (
    <Panel title="New handoff" eyebrow="Handoff controls">
      <Field label="Completed assignment">
        <select value={draft.source_assignment_id} onChange={event => onChange({ ...draft, source_assignment_id: event.target.value, to_agent_id: '' })}>
          <option value="">Choose completed work</option>
          {assignments.map(assignment => <option key={assignment.id} value={assignment.id}>{assignment.title}</option>)}
        </select>
      </Field>
      <Field label="Target agent">
        <select value={draft.to_agent_id} onChange={event => onChange({ ...draft, to_agent_id: event.target.value })}>
          <option value="">Choose the next specialist</option>
          {targets.map(agent => <option key={agent.id} value={agent.id}>{agent.name}</option>)}
        </select>
      </Field>
      <Field label="Artifact type">
        <select value={draft.artifact_type} onChange={event => onChange({ ...draft, artifact_type: event.target.value as HandoffArtifactType })}>
          {ARTIFACT_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
        </select>
      </Field>
      <Field label="Instructions"><textarea value={draft.instructions} onChange={event => onChange({ ...draft, instructions: event.target.value })} rows={5} placeholder="Turn this artifact into the next finished work product." /></Field>
      {source && <TextBlock label="Artifact preview" value={source.result} muted />}
      {error && <ErrorNotice error={error} />}
      <button disabled={!draft.source_assignment_id || !draft.to_agent_id || !draft.instructions.trim() || pending} onClick={onSubmit} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{pending ? 'Routing…' : 'Create handoff'}</button>
    </Panel>
  )
}

function HandoffDetails({ handoff, assignment, fromAgent, toAgent, isRunning, isRemoving, error, onRun, onRemove }: { handoff: Handoff; assignment: Assignment | null; fromAgent: string; toAgent: string; isRunning: boolean; isRemoving: boolean; error: Error | null; onRun: () => void; onRemove: () => void }) {
  const runnable = assignment?.status === 'queued' || assignment?.status === 'failed'
  return (
    <Panel title={`${fromAgent} → ${toAgent}`} eyebrow="Handoff route">
      <div className="grid grid-cols-2 gap-3"><Meta label="Artifact" value={handoff.artifact_type} /><Meta label="Status" value={handoff.status} /></div>
      <TextBlock label="Instructions" value={handoff.instructions} />
      <TextBlock label="Artifact snapshot" value={handoff.artifact} muted />
      {assignment?.result && <TextBlock label="Downstream work product" value={assignment.result} />}
      {assignment?.error && <div className="border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">{assignment.error}</div>}
      {error && <ErrorNotice error={error} />}
      {runnable && <button disabled={isRunning} onClick={onRun} className="w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{isRunning ? 'Mini is working…' : assignment?.status === 'failed' ? 'Retry downstream work' : 'Run downstream assignment'}</button>}
      <button disabled={isRemoving || handoff.status === 'running'} onClick={onRemove} className="w-full px-3 py-2 text-xs text-neutral-600 hover:text-red-300 disabled:opacity-30">Unlink handoff</button>
    </Panel>
  )
}

function Panel({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return <><div className="border-b border-neutral-900 px-6 py-5"><div className="text-[10px] font-semibold tracking-[0.24em] uppercase text-neutral-500">{eyebrow}</div><div className="mt-2 break-words text-lg text-white">{title}</div></div><div className="space-y-5 px-6 py-6">{children}</div></>
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="agent-field block"><span className="mb-2 block text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">{label}</span>{children}</label>
}

function Meta({ label, value }: { label: string; value: string }) {
  return <div className="border border-neutral-800 bg-neutral-900/60 p-3"><div className="text-[10px] uppercase tracking-[0.14em] text-neutral-600">{label}</div><div className="mt-1 break-words text-sm capitalize text-neutral-200">{value}</div></div>
}

function TextBlock({ label, value, muted = false }: { label: string; value: string; muted?: boolean }) {
  return <div><div className="mb-2 text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">{label}</div><div className={`max-h-72 overflow-y-auto whitespace-pre-wrap border border-neutral-800 bg-neutral-900/60 p-3 text-xs leading-relaxed ${muted ? 'text-neutral-500' : 'text-neutral-300'}`}>{value}</div></div>
}

function StatusLabel({ status }: { status: AssignmentStatus }) {
  const color = status === 'completed' ? 'border-green-800 text-green-300' : status === 'running' ? 'border-amber-700 text-amber-300' : status === 'failed' ? 'border-red-900 text-red-300' : 'border-neutral-700 text-neutral-400'
  return <span className={`shrink-0 border px-2 py-1 text-[10px] uppercase tracking-[0.12em] ${color}`}>{status}</span>
}

function ErrorNotice({ error }: { error: Error }) {
  return <div role="alert" className="border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">{error.message}</div>
}

function relativeTime(value: string) {
  const elapsed = Math.max(0, Date.now() - new Date(value).getTime())
  const minutes = Math.floor(elapsed / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}
