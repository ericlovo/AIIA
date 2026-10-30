import { useDecisionQueue } from './useDecisionQueue'
import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Plus, RotateCcw, X } from 'lucide-react'
import { groupWork, isTestFixture, workLane, type WorkLane } from './workQueue'
import { AssignmentHistory } from './AssignmentHistory'
import { RoutingAdvisor } from './RoutingAdvisor'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type Agent,
  type Assignment,
  type AssignmentDefinition,
  type AssignmentPriority,
  type AssignmentStatus,
  type ReviewStatus,
  type Handoff,
  type HandoffArtifactType,
  type HandoffDefinition,
  type GitWrite,
  type GitWorkspace,
  type RepositoryResource,
} from '../lib/api'
import { AgentPicker } from './AgentPicker'
import { assignmentLabel, assignmentOrigin, assignmentNextAction, reviewLabel } from './assignmentReview'
import { formatRoute, type StudioView } from './studioRoute'
import { PageHeader } from './PageHeader'
import { replaceRoute } from './useStudioRoute'

const EMPTY_ASSIGNMENTS: Assignment[] = []
const EMPTY_HANDOFFS: Handoff[] = []
const EMPTY_WORKSPACES: GitWorkspace[] = []
const EMPTY_WRITES: GitWrite[] = []
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
  view: Extract<StudioView, 'assignments' | 'handoffs'>
  onRouteHandoff: (sourceId: string, targetId: string) => void
  initialAgentId?: string
  initialAssignmentId?: string
  initialHandoffSourceId?: string
  initialHandoffTargetId?: string
}

export function WorkBoard({
  agents,
  view,
  onRouteHandoff,
  initialAgentId = '',
  initialAssignmentId = '',
  initialHandoffSourceId = '',
  initialHandoffTargetId = '',
}: WorkBoardProps) {
  const inspectorRef = useRef<HTMLElement>(null)
  const newWorkRef = useRef<HTMLButtonElement>(null)
  const qc = useQueryClient()
  const decisionQueue = useDecisionQueue()
  const { data: assignmentData, isLoading: assignmentsLoading, isError: assignmentsError } = useQuery({
    queryKey: view === 'assignments' ? ['assignments', 'summary'] : ['assignments'],
    queryFn: view === 'assignments' ? api.assignmentSummaries : api.assignments,
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
  const { data: writeData } = useQuery({
    queryKey: ['git-writes'],
    queryFn: () => api.gitWrites(),
    refetchInterval: 5_000,
  })
  const { data: resourceData } = useQuery({
    queryKey: ['agent-resources'],
    queryFn: api.agentResources,
  })
  const [previewOpen, setPreviewOpen] = useState(false)
  const fixturePreview = useQuery({ queryKey: ['fixture-preview'], queryFn: api.fixturePreview, enabled: previewOpen, refetchInterval: previewOpen ? 5000 : false })
  const attention = useMutation({
    mutationFn: ({ item, active }: { item: Assignment; active: boolean }) => api.acknowledgeAttention(item, active),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['assignments'] }),
  })
  const assignments = assignmentData?.assignments ?? EMPTY_ASSIGNMENTS
  const handoffs = handoffData?.handoffs ?? EMPTY_HANDOFFS
  const workspaces = workspaceData?.workspaces ?? EMPTY_WORKSPACES
  const writes = writeData?.writes ?? EMPTY_WRITES
  const repos = resourceData?.repos ?? EMPTY_REPOS
  const [visibleGroups, setVisibleGroups] = useState(50)
  const [lane, setLane] = useState<WorkLane>('needs-you')
  const [selectedAssignmentId, setSelectedAssignmentId] = useState<string | null>(initialAssignmentId || null)
  const [creatingAssignment, setCreatingAssignment] = useState(Boolean(initialAgentId) && !initialAssignmentId)
  const [selectedHandoffId, setSelectedHandoffId] = useState<string | null>(null)
  // Selecting inside the view rewrites the address without a history entry, so
  // a selected assignment can be copied or reloaded. The first render is the
  // arrival itself, which already matches the address, so it is left alone.
  const syncedSelection = useRef(selectedAssignmentId)
  useEffect(() => {
    if (view !== 'assignments' || syncedSelection.current === selectedAssignmentId) return
    syncedSelection.current = selectedAssignmentId
    replaceRoute(selectedAssignmentId ? { view: 'assignments', assignmentId: selectedAssignmentId } : { view: 'assignments' })
  }, [view, selectedAssignmentId])
  const [assignmentDraft, setAssignmentDraft] = useState<AssignmentDefinition>(() => ({
    ...EMPTY_ASSIGNMENT,
    agent_id: initialAgentId,
  }))
  const [handoffDraft, setHandoffDraft] = useState<HandoffDefinition>(() => ({
    ...EMPTY_HANDOFF,
    source_assignment_id: initialHandoffSourceId,
    to_agent_id: initialHandoffTargetId,
  }))
  const selectedRow = assignments.find(item => item.id === selectedAssignmentId)
  const detail = useQuery({
    queryKey: ['assignments', 'detail', selectedAssignmentId],
    queryFn: () => api.assignment(selectedAssignmentId!),
    enabled: Boolean(selectedAssignmentId && (!selectedRow || selectedRow.summary)),
    refetchInterval: 5_000,
  })
  const selectedAssignment = selectedRow && !selectedRow.summary ? selectedRow : detail.data?.assignment ?? null
  const selectedHandoff = handoffs.find(item => item.id === selectedHandoffId) ?? null
  const inspectedAssignmentId = selectedAssignment?.id
  useEffect(() => {
    if (view !== 'assignments' || (!creatingAssignment && !inspectedAssignmentId)) return
    inspectorRef.current?.focus({ preventScroll: true })
    if (window.matchMedia('(max-width:1023px)').matches) {
      inspectorRef.current?.scrollIntoView({ block: 'start' })
    }
  }, [view, creatingAssignment, inspectedAssignmentId])

  const createAssignment = useMutation({
    mutationFn: api.createAssignment,
    onSuccess: ({ assignment }) => {
      setSelectedAssignmentId(assignment.id)
      setCreatingAssignment(false)
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
  const approveWrite = useMutation({
    mutationFn: api.approveGitWrite,
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['git-writes'] })
      qc.invalidateQueries({ queryKey: ['git-workspaces'] })
    },
  })
  const rejectWrite = useMutation({
    mutationFn: (writeId: string) => api.rejectGitWrite(writeId),
    onSettled: () => qc.invalidateQueries({ queryKey: ['git-writes'] }),
  })

  const completedAssignments = assignments.filter(item => item.status === 'completed' && (item.has_result ?? Boolean(item.result)))
  const runningCount = assignments.filter(item => item.status === 'running').length
  const manualStartCount = assignments.filter(item => item.status === 'queued').length

  function agentName(agentId: string) {
    return agents.find(agent => agent.id === agentId)?.name ?? 'Removed agent'
  }

  function prepareHandoff(assignment: Assignment) {
    onRouteHandoff(assignment.id, '')
  }

  function openNewWork() {
    setSelectedAssignmentId(null)
    setCreatingAssignment(true)
    createAssignment.reset()
  }

  function closeNewWork() {
    setCreatingAssignment(false)
    replaceRoute({ view: 'assignments' })
    newWorkRef.current?.focus()
  }

  function openAssignment(id: string) {
    setCreatingAssignment(false)
    runAssignment.reset()
    removeAssignment.reset()
    setSelectedAssignmentId(id)
  }

  const pendingApprovalIds = decisionQueue.pendingApprovalIds
  const matchingEvidence = selectedAssignment?.attention_key ? assignments.filter(item => item.attention_key === selectedAssignment.attention_key) : []
  const laneItems = assignments.filter(item => workLane(item, pendingApprovalIds.has(item.id)) === lane)
  const groups = view === 'assignments' && lane === 'needs-you' ? decisionQueue.groups : groupWork(laneItems)

  const loading = view === 'assignments' ? assignmentsLoading : handoffsLoading
  const items = view === 'assignments' ? assignments : handoffs
  const inspectorOpen = view === 'handoffs' || creatingAssignment || Boolean(selectedAssignmentId)

  return (
    <main className={`grid h-full min-h-0 max-h-full auto-rows-max grid-cols-1 overflow-y-auto bg-neutral-950 lg:grid-rows-[minmax(0,1fr)] lg:overflow-hidden [&_button]:min-h-11 [&_button]:min-w-11 [&_summary]:min-h-11 [&_summary]:tracking-normal ${inspectorOpen ? 'lg:grid-cols-[minmax(0,1fr)_minmax(390px,480px)]' : ''}`}>
      <section className={`flex min-w-0 flex-col border-neutral-900 lg:overflow-hidden ${inspectorOpen ? 'border-b lg:border-r lg:border-b-0' : ''}`}>
        <PageHeader title={view === 'assignments' ? 'Work' : 'Handoff ledger'} meta={<>
          <span>{assignments.length} assignments</span>
          <span>{manualStartCount} awaiting manual start</span>
          <span>{runningCount} running</span>
          {view === 'handoffs' && <span>{handoffs.length} handoffs</span>}
        </>} />

        <div className="overflow-y-auto px-5 py-6 sm:px-7 lg:min-h-0 lg:flex-1">
          <div className="mx-auto max-w-5xl">
            <div className="mb-4 flex items-center justify-between gap-4">
              <div className="text-xs font-semibold text-neutral-400">
                {view === 'assignments' ? `${groups.length} ${groups.length === 1 ? 'group' : 'groups'} · ${laneItems.length} ${laneItems.length === 1 ? 'record' : 'records'}` : 'Artifact routes'}
              </div>
              <button
                ref={newWorkRef}
                onClick={() => view === 'assignments' ? openNewWork() : setSelectedHandoffId(null)}
                aria-expanded={view === 'assignments' ? creatingAssignment : undefined}
                aria-controls={view === 'assignments' && creatingAssignment ? 'new-work-form' : undefined}
                className="inline-flex shrink-0 items-center gap-2 border border-neutral-700 px-3 py-2 text-sm text-neutral-200 hover:border-cyan-500/60 hover:text-cyan-200"
              >
                <Plus size={16} aria-hidden="true" />{view === 'assignments' ? 'New work' : 'New handoff'}
              </button>
            </div>

            {view === 'assignments' && <p className="mt-4 text-sm leading-relaxed text-neutral-300">Start with Needs you. Open a record to see the decision and its evidence. Reviewing output never grants Git permissions. History keeps completed reviews and dismissed records.</p>}
            {view === 'assignments' && <section className="my-4 space-y-3 border border-neutral-800 p-3 text-sm">
              <button onClick={() => setPreviewOpen(!previewOpen)} aria-expanded={previewOpen}>Preview suspected test records</button>
              {previewOpen && <div className="space-y-3">
                {fixturePreview.isLoading && <p>Loading preview…</p>}
                {fixturePreview.error && <p role="alert">Could not load classification preview.</p>}
                {fixturePreview.data && <>
                  <p>{fixturePreview.data.candidates.length} suspected test records · {fixturePreview.data.blocked.length} blocked by pending approvals</p>
                  <p>{fixturePreview.data.warning}</p>
                  <p>Preview only. Original records, verdicts and history remain unchanged. No classification action is enabled.</p>
                  <details><summary>Inspect candidate IDs and evidence</summary>
                    {fixturePreview.data.candidates.map(item => <div key={item.id} className="break-words border-t border-neutral-800 py-3"><button onClick={() => openAssignment(item.id)}>{item.title} · {item.id}</button><p>{item.result}</p><p>{item.reason}</p><p className="break-all text-xs">Fingerprint: {item.fingerprint}</p></div>)}
                  </details>
                  {fixturePreview.data.blocked.map(item => <p key={item.id}>Approval protected: {item.title} · {item.id}</p>)}
                </>}
              </div>}
            </section>}

            {view === 'assignments' && decisionQueue.error && <p role="alert" className="py-3 text-sm text-red-300">Could not refresh all decisions or permissions. Counts may be stale; no all-clear is available.</p>}
            {assignmentsError && <p role="alert" className="py-6 text-sm text-red-300">Could not refresh the backlog. Counts may be stale; refresh before deciding.</p>}
            {loading && <p role="status" className="py-6 text-sm text-neutral-400">Loading work...</p>}
            {!loading && !assignmentsError && items.length === 0 && (
              <button
                onClick={() => view === 'assignments' ? openNewWork() : setSelectedHandoffId(null)}
                className="min-h-48 w-full border border-dashed border-neutral-700 p-6 text-left hover:border-cyan-500/50"
              >
                <div className="text-base text-white">{view === 'assignments' ? 'No assignments yet' : 'No handoffs yet'}</div>
                <div className="mt-2 text-xs text-neutral-500">{view === 'assignments' ? 'Create focused work for one local agent.' : 'Route a completed artifact to the next specialist.'}</div>
              </button>
            )}

            {view === 'assignments' && <nav aria-label="Work views" className="mb-5 flex flex-wrap gap-2">
              {([['needs-you', 'Needs you'], ['working', 'Working'], ['history', 'History']] as const).map(([value, label]) => <button key={value} aria-pressed={lane === value} onClick={() => { setLane(value); setVisibleGroups(50) }} className={`border px-3 py-2 text-sm ${lane === value ? 'border-cyan-400 text-cyan-200' : 'border-neutral-700 text-neutral-300'}`}>
                {label} ({value === 'needs-you' ? decisionQueue.error ? 'Unavailable' : decisionQueue.loading ? 'Loading' : decisionQueue.total : assignments.filter(item => workLane(item, pendingApprovalIds.has(item.id)) === value).length})
              </button>)}
            </nav>}
            {view === 'assignments' && lane === 'needs-you' && decisionQueue.savedIdeas > 0 && <p className="mb-4 text-sm text-neutral-300"><a className="inline-flex min-h-11 items-center text-cyan-200 underline" href={formatRoute({ view: 'inbox', source: 'all' })}>Review saved ideas ({decisionQueue.savedIdeas})</a> · Decide what to keep as memory. This does not approve execution.</p>}
            {view === 'assignments' && lane === 'needs-you' && decisionQueue.missingApprovalIds.length > 0 && <p role="alert" className="mb-4 text-sm text-amber-200">Permission requests reference unavailable work: {decisionQueue.missingApprovalIds.join(', ')}. These approvals remain pending; do not proceed until their records are available.</p>}
            {view === 'assignments' ? (
              <div className="space-y-3">
                {!loading && !assignmentsError && !decisionQueue.error && !decisionQueue.loading && laneItems.length === 0 && (lane !== 'needs-you' || decisionQueue.total === 0) && <p role="status" className="text-sm text-neutral-400">{lane === 'needs-you' ? 'Nothing needs your decision. Working and History remain available.' : 'No work in this view.'}</p>}
                {groups.slice(0, visibleGroups).map(group => <section key={group[0].id} className="border border-neutral-800">
                  <AssignmentCard approvalPending={group.some(item => pendingApprovalIds.has(item.id))} assignment={group[0]} agentName={agentName(group[0].agent_id)} selected={selectedAssignmentId === group[0].id} onSelect={() => openAssignment(group[0].id)} />
                  {group.length > 1 && <details className="px-4 pb-3 text-sm text-neutral-300"><summary className="cursor-pointer">{group.length} matching records · Review each separately</summary>
                    <p className="py-2 text-xs text-neutral-400">Matching task text. Owners, sources, evidence and decisions remain separate for each run.</p>
                    {group.map(item => <button key={item.id} onClick={() => openAssignment(item.id)} className="block w-full border-t border-neutral-800 py-2 text-left">Open {new Date(item.updated_at).toLocaleString()} · {item.id} · {agentName(item.agent_id)} · {reviewLabel(item)}</button>)}
                  </details>}
                </section>)}
                {groups.length > visibleGroups && <button className="border border-neutral-700 px-4 py-2 text-sm text-neutral-200" onClick={() => setVisibleGroups(count => count + 50)}>Show more groups ({groups.length - visibleGroups} remaining)</button>}
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

      {inspectorOpen && <aside ref={inspectorRef} tabIndex={-1} aria-label={view === 'assignments' ? creatingAssignment ? 'New work' : 'Selected work' : 'Handoff details'} className="min-h-0 min-w-0 bg-neutral-950 outline-none lg:overflow-y-auto">
        {view === 'assignments' && selectedAssignment?.attention_key && !selectedAssignment.dismissed_at && <section aria-label="Attention decision" className="m-4 space-y-3 border border-neutral-700 p-4 text-sm">
          <h3>{selectedAssignment.attention_seen ? 'Acknowledged repeat' : 'Attention decision'}</h3>
          <p>{matchingEvidence.length} record(s) have identical evidence and scope from the same agent. Acknowledgement moves these and future exact repeats to History. It does not accept output, start work, or approve permissions. Changed evidence returns to Needs you.</p>
          {matchingEvidence.some(item => pendingApprovalIds.has(item.id)) && <p>Pending approval must be decided separately. Acknowledgement is blocked.</p>}
          <button className="border border-neutral-600 p-3 disabled:opacity-50" disabled={attention.isPending || (!selectedAssignment.attention_seen && matchingEvidence.some(item => pendingApprovalIds.has(item.id)))} onClick={() => attention.mutate({ item: selectedAssignment, active: !selectedAssignment.attention_seen })}>{selectedAssignment.attention_seen ? 'Restore identical reports to attention' : 'Acknowledge identical reports'}</button>
          {attention.error && <p role="alert">{attention.error.message}</p>}
        </section>}
        {view === 'assignments' ? (
          selectedAssignment ? (
            <AssignmentDetails
              onOpenAssignment={openAssignment}
              assignment={selectedAssignment}
              agent={agents.find(item => item.id === selectedAssignment.agent_id) ?? null}
              agentName={agentName(selectedAssignment.agent_id)}
              workspace={workspaces.find(item => item.assignment_id === selectedAssignment.id) ?? null}
              writes={writes}
              repo={repos.find(item => item.id === agents.find(agent => agent.id === selectedAssignment.agent_id)?.repo_id) ?? null}
              isRunning={runAssignment.isPending}
              isRemoving={removeAssignment.isPending}
              isRequestingWorkspace={requestWorkspace.isPending}
              isApprovingWorkspace={approveWorkspace.isPending}
              isApprovingWrite={approveWrite.isPending}
              isRejectingWrite={rejectWrite.isPending}
              hasHandoff={handoffs.some(item => item.source_assignment_id === selectedAssignment.id || item.target_assignment_id === selectedAssignment.id)}
              error={runAssignment.error ?? removeAssignment.error ?? requestWorkspace.error ?? approveWorkspace.error ?? approveWrite.error ?? rejectWrite.error}
              onRun={() => runAssignment.mutate(selectedAssignment.id)}
              onHandoff={() => prepareHandoff(selectedAssignment)}
              onRequestWorkspace={() => requestWorkspace.mutate(selectedAssignment.id)}
              onApproveWorkspace={workspaceId => approveWorkspace.mutate(workspaceId)}
              onApproveWrite={writeId => approveWrite.mutate(writeId)}
              onRejectWrite={writeId => rejectWrite.mutate(writeId)}
              onRemove={() => removeAssignment.mutate(selectedAssignment.id)}
            />
          ) : creatingAssignment ? (
            <AssignmentForm
              agents={agents}
              draft={assignmentDraft}
              pending={createAssignment.isPending}
              error={createAssignment.error}
              onChange={setAssignmentDraft}
              onSubmit={() => createAssignment.mutate(assignmentDraft)}
              onClose={closeNewWork}
            />
          ) : (
            <p role="status" className="p-6 text-sm text-neutral-400">{assignmentsLoading || detail.isLoading ? 'Loading assignment...' : detail.isError ? 'Could not load evidence. Refresh to try again.' : 'This assignment is not available.'}</p>
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
      </aside>}
    </main>
  )
}

function AssignmentCard({ assignment, agentName, selected, onSelect, approvalPending }: { approvalPending: boolean; assignment: Assignment; agentName: string; selected: boolean; onSelect: () => void }) {
  return (
    <button onClick={onSelect} aria-pressed={selected} className={`w-full min-h-40 min-w-0 border p-5 text-left transition-colors ${selected ? 'border-cyan-400/70 bg-cyan-500/10' : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-600'}`}>
      <div className="mb-2 text-xs text-cyan-200">{approvalPending && 'Approval needed · '}{assignment.attention_seen && 'Acknowledged repeat · '}{isTestFixture(assignment) ? 'Test fixture' : assignmentOrigin(assignment)}{assignmentLabel(assignment) ? ` · ${assignmentLabel(assignment)}` : ''}</div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="break-words text-base font-medium text-white">{assignment.title}</div>
          <div className="mt-1 break-words text-xs text-cyan-300/80">{agentName}</div>
        </div>
        <StatusLabel status={assignment.status} />
      </div>
      <p className="mt-4 line-clamp-2 break-words text-sm leading-relaxed text-neutral-400">{assignment.objective}</p>
      <div className="mt-5 flex flex-wrap items-center justify-between gap-2 text-xs text-neutral-400">
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
      <div className="mt-3 text-xs text-cyan-300/70">{handoff.artifact_type}</div>
      <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-neutral-500">{handoff.instructions}</p>
    </button>
  )
}

function OutputRecovery({ assignment, busy }: { assignment: Assignment; busy: boolean }) {
  const qc = useQueryClient()
  const recovery = useMutation({
    mutationFn: () => api.recoverAssignment(assignment.id),
    retry: false,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['assignments'] })
      qc.invalidateQueries({ queryKey: ['handoffs'] })
    },
  })
  if (!assignment.recovery_pending) {
    return assignment.recovered_at ? <div role="status" className="text-xs text-emerald-300">Saved output recovered</div> : null
  }
  return <div className="space-y-2 border-t border-neutral-800 pt-3">
    <div className="text-xs text-amber-300">Output reconciliation pending</div>
    <button onClick={() => recovery.mutate()} disabled={busy || recovery.isPending} className="flex w-full items-center justify-center gap-2 border border-neutral-700 px-3 py-2 text-sm text-white disabled:opacity-40">
      <RotateCcw size={14} aria-hidden="true" />{recovery.isPending ? 'Recovering output…' : 'Recover saved output'}
    </button>
    {recovery.error && <ErrorNotice error={recovery.error} />}
  </div>
}

function AssignmentForm({ agents, draft, pending, error, onChange, onSubmit, onClose }: { agents: Agent[]; draft: AssignmentDefinition; pending: boolean; error: Error | null; onChange: (draft: AssignmentDefinition) => void; onSubmit: () => void; onClose: () => void }) {
  return (
    <div id="new-work-form">
    <Panel title="New assignment" eyebrow="New work" actions={<button type="button" onClick={onClose} disabled={pending} aria-label="Close new work" title="Close new work" className="inline-flex items-center justify-center text-neutral-400 hover:text-white disabled:opacity-40"><X size={18} aria-hidden="true" /></button>}>
      <Field label="Title"><input value={draft.title} onChange={event => onChange({ ...draft, title: event.target.value })} placeholder="Map the authorization surface" /></Field>
      <Field label="Assigned agent">
        <AgentPicker agents={agents} value={draft.agent_id} onChange={agent_id => onChange({ ...draft, agent_id })} placeholder="Choose an agent" />
      </Field>
      <Field label="Objective"><textarea value={draft.objective} onChange={event => onChange({ ...draft, objective: event.target.value })} rows={5} placeholder="Return the five highest-leverage integration points." /></Field>
      <Disclosure title="Context, success criteria and priority">
        <Field label="Context"><textarea value={draft.context} onChange={event => onChange({ ...draft, context: event.target.value })} rows={4} placeholder="Relevant decisions, constraints, or source material." /></Field>
        <Field label="Success criteria"><textarea value={draft.success_criteria} onChange={event => onChange({ ...draft, success_criteria: event.target.value })} rows={3} placeholder="What must be true for this work to be done?" /></Field>
        <Field label="Priority">
          <select value={draft.priority} onChange={event => onChange({ ...draft, priority: event.target.value as AssignmentPriority })}>
            {PRIORITIES.map(priority => <option key={priority} value={priority}>{priority}</option>)}
          </select>
        </Field>
      </Disclosure>
      <Disclosure title="Routing advice (optional)">
        <RoutingAdvisor agents={agents} onSelect={agent_id => onChange({ ...draft, agent_id })} />
      </Disclosure>
      {error && <ErrorNotice error={error} />}
      <p className="text-sm leading-relaxed text-neutral-400">Creating saves this assignment as awaiting manual start. No agent runs until you choose Run assignment.</p>
      <button disabled={!draft.title.trim() || !draft.objective.trim() || !draft.agent_id || pending} onClick={onSubmit} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{pending ? 'Creating…' : 'Create assignment'}</button>
    </Panel>
    </div>
  )
}

interface AssignmentDetailsProps {
  onOpenAssignment: (id: string) => void
  assignment: Assignment
  agent: Agent | null
  agentName: string
  workspace: GitWorkspace | null
  writes: GitWrite[]
  repo: RepositoryResource | null
  isRunning: boolean
  isRemoving: boolean
  isRequestingWorkspace: boolean
  isApprovingWorkspace: boolean
  isApprovingWrite: boolean
  isRejectingWrite: boolean
  hasHandoff: boolean
  error: Error | null
  onRun: () => void
  onHandoff: () => void
  onRequestWorkspace: () => void
  onApproveWorkspace: (workspaceId: string) => void
  onApproveWrite: (writeId: string) => void
  onRejectWrite: (writeId: string) => void
  onRemove: () => void
}

function AssignmentDetails({ assignment, agent, agentName, workspace, writes, repo, isRunning, isRemoving, isRequestingWorkspace, isApprovingWorkspace, isApprovingWrite, isRejectingWrite, hasHandoff, error, onRun, onHandoff, onRequestWorkspace, onApproveWorkspace, onApproveWrite, onRejectWrite, onRemove, onOpenAssignment }: AssignmentDetailsProps) {
  const runnable = assignment.status === 'queued' || assignment.status === 'failed'
  const gitEnabled = agent?.tools.includes('Git workspace') ?? false
  const workspaceWrites = workspace ? writes.filter(item => item.workspace_id === workspace.id) : []
  const pendingWriteCount = workspaceWrites.filter(item => item.status === 'pending').length
  const workspaceNeedsAttention = workspace?.status === 'pending' || workspace?.status === 'failed' || pendingWriteCount > 0
  return (
    <Panel title={assignment.title} eyebrow="Decision and evidence">
      {isTestFixture(assignment) && <p className="text-sm text-amber-200">Test fixture · Preserved for reference. Any pending approval still needs your decision.</p>}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="min-w-0 break-words text-sm text-neutral-300">{agentName}</span>
        <StatusLabel status={assignment.status} />
      </div>
      <section aria-label="Next action" className="space-y-2 border-l-2 border-cyan-600 pl-3">
        <h3 className="text-xs font-semibold text-cyan-200">Next action</h3>
        <p className="text-sm leading-relaxed text-neutral-300">{assignmentNextAction(assignment, workspace?.status === 'pending', pendingWriteCount, isTestFixture(assignment))}</p>
      </section>
      {error && <ErrorNotice error={error} />}
      {runnable && !assignment.dismissed_at && <button disabled={isRunning} onClick={onRun} className="min-h-11 w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{isRunning ? 'Mini is working…' : assignment.status === 'failed' ? 'Retry assignment' : 'Run assignment'}</button>}
      {assignment.error && <section aria-label="Failure evidence" className="border-l-2 border-red-500 bg-red-950/20 p-3 text-sm text-red-200"><h3 className="font-medium">What went wrong</h3><p className="mt-2 whitespace-pre-wrap break-words">{assignment.error}</p></section>}
      {assignment.result && <TextBlock label="Work product" value={assignment.result} />}
      {assignment.status === 'completed' && assignment.result.trim() && (assignment.dismissed_at || assignment.attention_seen
        ? <Disclosure key={`optional-review-${assignment.id}`} title="Output verdict and optional review"><ArtifactReview key={assignment.id} assignment={assignment} /></Disclosure>
        : <ArtifactReview key={assignment.id} assignment={assignment} />)}
      <Disclosure key={`brief-${assignment.id}`} title="Assignment brief" initiallyOpen>
        <TextBlock label="Objective" value={assignment.objective} />
        {assignment.success_criteria && <TextBlock label="Success criteria" value={assignment.success_criteria} />}
        {assignment.context && <TextBlock label="Context" value={assignment.context} muted />}
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <div><dt className="text-xs text-neutral-400">Priority</dt><dd className="mt-1 capitalize text-neutral-200">{assignment.priority}</dd></div>
          <div><dt className="text-xs text-neutral-400">Started by</dt><dd className="mt-1 text-neutral-200">{assignmentOrigin(assignment)}</dd></div>
        </dl>
        {assignment.revision_of && <button onClick={() => onOpenAssignment(assignment.revision_of!)} className="text-sm text-cyan-200 underline">Open original assignment</button>}
      </Disclosure>
      {assignment.review_status === 'rejected' && <RevisionPanel key={`revision-${assignment.id}`} assignment={assignment} onOpen={onOpenAssignment} />}
      <OutputRecovery key={`recovery-${assignment.id}`} assignment={assignment} busy={isRunning} />
      {(assignment.status === 'completed' || assignment.status === 'failed') && <DismissalPanel key={`dismiss-${assignment.id}`} assignment={assignment} />}
      <Disclosure key={`history-${assignment.id}`} title="Attempt history">
        <AssignmentHistory assignmentId={assignment.id} />
      </Disclosure>
      {assignment.status === 'completed' && (
        <Disclosure key={`git-${assignment.id}-${workspace?.status}-${pendingWriteCount}`} title={workspace?.status === 'pending' ? 'Git workspace: approval needed' : pendingWriteCount ? `Git workspace: ${pendingWriteCount} pending writes` : workspace?.status === 'failed' ? 'Git workspace: setup failed' : 'Git workspace'} initiallyOpen={workspaceNeedsAttention}>
        <GitWorkspacePanel
          agentName={agentName}
          gitEnabled={gitEnabled}
          repo={repo}
          workspace={workspace}
          writes={workspaceWrites}
          isRequesting={isRequestingWorkspace}
          isApproving={isApprovingWorkspace}
          isApprovingWrite={isApprovingWrite}
          isRejectingWrite={isRejectingWrite}
          onRequest={onRequestWorkspace}
          onApprove={onApproveWorkspace}
          onApproveWrite={onApproveWrite}
          onRejectWrite={onRejectWrite}
        />
        </Disclosure>
      )}
      <Disclosure key={`actions-${assignment.id}`} title="Handoff and record actions">
        {assignment.status === 'completed' && <button onClick={onHandoff} className="w-full border border-cyan-700 px-3 py-2.5 text-sm font-medium text-cyan-200">Hand off work</button>}
        <button disabled={isRemoving || assignment.status === 'running' || hasHandoff} onClick={onRemove} title={hasHandoff ? 'Unlink the handoff before deleting connected work' : undefined} className="w-full px-3 py-2 text-xs text-neutral-400 hover:text-red-300 disabled:opacity-30">Delete assignment</button>
      </Disclosure>
      <a href={formatRoute({ view: 'assignments' })} className="inline-flex min-h-11 items-center text-sm text-cyan-200 underline">Back to decision queue</a>
    </Panel>
  )
}

function RevisionPanel({ assignment, onOpen }: { assignment: Assignment; onOpen: (id: string) => void }) {
  const qc = useQueryClient()
  const [note, setNote] = useState(assignment.review_note || '')
  const revision = useMutation({
    mutationFn: () => api.reviseAssignment(assignment.id, assignment.review_version!, note),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['assignments'] }),
  })
  const existingRevisionId = assignment.revision_ids?.at(-1)
  return <section aria-label="Request revision" className="space-y-3 border border-neutral-700 p-3">
    <p className="text-xs leading-relaxed text-neutral-400">Create linked follow-up work for the same agent. This preserves the rejected artifact and does not run the revision automatically.</p>
    {existingRevisionId ? <button className="w-full border border-cyan-700 px-3 py-2 text-sm text-cyan-200" onClick={() => onOpen(existingRevisionId)}>Open queued revision</button> : <>
    <Field label="Revision feedback"><textarea value={note} onChange={event => setNote(event.target.value)} maxLength={2000} rows={3} /></Field>
    <button disabled={!note.trim() || !assignment.review_version || revision.isPending || revision.isSuccess} onClick={() => revision.mutate()} className="flex w-full items-center justify-center gap-2 border border-cyan-700 px-3 py-2 text-sm text-cyan-200 disabled:opacity-40"><RotateCcw size={14} />{revision.isPending ? 'Creating revision...' : 'Request revision'}</button>
    {revision.data && <div role="status" className="text-xs text-cyan-200">Revision queued. <button className="underline" onClick={() => onOpen(revision.data.assignment.id)}>Open revision</button></div>}
    {revision.error && <ErrorNotice error={new Error(revisionError(revision.error.message))} />}
    </>}
  </section>
}

function DismissalPanel({ assignment }: { assignment: Assignment }) {
  const qc = useQueryClient()
  const dismissed = Boolean(assignment.dismissed_at)
  const [note, setNote] = useState('')
  const dismiss = useMutation({
    mutationFn: (next: boolean) => api.dismissAssignment(assignment.id, next, assignment.review_version!, note),
    onSuccess: ({ assignment: updated }) => {
      qc.setQueryData<{ assignments: Assignment[] }>(['assignments'], previous => previous ? {
        assignments: previous.assignments.map(item => item.id === updated.id ? updated : item),
      } : previous)
      setNote('')
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ['assignments'] }),
  })
  return <section aria-label="Attention tracking" className="space-y-3 border border-neutral-700 p-3">
    <div className="text-sm text-cyan-200">{dismissed ? 'Dismissed' : 'Not dismissed'}</div>
    <p className="text-xs leading-relaxed text-neutral-400">{dismissed
      ? 'This assignment is hidden from the attention list. Its outcome and any review verdict are unchanged; restoring brings it back with both intact.'
      : 'Dismissing hides this from the attention list without judging the work. The outcome, the failure and any review verdict are all kept.'}</p>
    {dismissed && assignment.dismissed_at && <p className="text-xs text-neutral-400">Dismissed {new Date(assignment.dismissed_at).toLocaleString()}</p>}
    {dismissed && assignment.dismiss_note && <p className="text-xs break-words text-neutral-400">{assignment.dismiss_note}</p>}
    {!dismissed && <Field label="Dismissal reason"><textarea value={note} onChange={event => setNote(event.target.value)} maxLength={2000} rows={2} placeholder="Why this no longer needs attention" /></Field>}
    <button type="button" disabled={dismiss.isPending || !assignment.review_version} onClick={() => dismiss.mutate(!dismissed)} className="border border-neutral-600 px-3 py-2 text-xs text-neutral-200 hover:border-cyan-300 disabled:opacity-40">{dismissed ? 'Restore to attention' : 'Dismiss'}</button>
    {dismiss.isPending && <p role="status" className="text-xs text-neutral-400">Saving...</p>}
    {dismiss.error && <ErrorNotice error={new Error(dismiss.error.message.includes('review_changed_refresh_required') ? 'The assignment changed. Reload it before dismissing again.' : dismiss.error.message.includes('assignment_not_settled') ? 'This assignment is still queued or running. Wait for it to finish.' : dismiss.error.message)} />}
  </section>
}

const REVIEW_ACTION: Record<ReviewStatus, string> = {
  accepted: 'Accept output',
  rejected: 'Reject output',
  unreviewed: 'Reopen review',
}

function ArtifactReview({ assignment }: { assignment: Assignment }) {
  const qc = useQueryClient()
  const [draftNote, setDraftNote] = useState({ version: assignment.review_version, note: assignment.review_note ?? '' })
  const note = draftNote.version === assignment.review_version ? draftNote.note : assignment.review_note ?? ''
  const review = useMutation({
    mutationFn: (decision: ReviewStatus) => api.reviewAssignment(assignment.id, decision, assignment.review_version!, note),
    onSuccess: ({ assignment: updated }) => {
      qc.setQueryData<{ assignments: Assignment[] }>(['assignments'], previous => previous ? {
        assignments: previous.assignments.map(item => item.id === updated.id ? updated : item),
      } : previous)
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ['assignments'] }),
  })
  const decision = assignment.review_status ?? 'unreviewed'
  const options = (['accepted', 'rejected', 'unreviewed'] as ReviewStatus[])
  return <section aria-label="Artifact review" className="space-y-3 border border-neutral-700 p-3">
    <div className="text-sm text-cyan-200">{reviewLabel(assignment)}</div>
    <p className="text-xs leading-relaxed text-neutral-400">Inspect the work product against the objective and success criteria. This decision records output quality only. It does not authorize execution, Git writes, deployment, or outreach. It is kept if the assignment is later dismissed.</p>
    {assignment.reviewed_at && <p className="text-xs text-neutral-400">Decision saved {new Date(assignment.reviewed_at).toLocaleString()}</p>}
    <Field label="Review note"><textarea value={note} onChange={event => setDraftNote({ version: assignment.review_version, note: event.target.value })} maxLength={2000} rows={3} placeholder="Evidence, gaps, or requested corrections" /></Field>
    {!assignment.review_version && <p role="alert" className="text-xs text-amber-200">Review is unavailable until the server supports artifact review.</p>}
    <div className="flex flex-wrap gap-2">{options.filter(value => value !== decision).map(value => <button key={value} disabled={review.isPending || !assignment.review_version} onClick={() => review.mutate(value)} className="border border-neutral-600 px-3 py-2 text-xs text-neutral-200 hover:border-cyan-300 disabled:opacity-40">{REVIEW_ACTION[value]}</button>)}</div>
    {review.isPending && <p role="status" className="text-xs text-neutral-400">Saving review...</p>}
    {review.error && <ErrorNotice error={new Error(review.error.message.includes('review_changed_refresh_required') ? 'The output or review changed. Reload the assignment and inspect it before deciding again.' : review.error.message.includes('assignment_not_settled') ? 'This assignment is still queued or running. Wait for it to finish before dismissing or reopening it.' : review.error.message.includes('review_persistence_failed') ? 'Review was not saved. Your previous decision is unchanged; retry when storage is available.' : review.error.message)} />}
  </section>
}

function GitWorkspacePanel({ agentName, gitEnabled, repo, workspace, writes, isRequesting, isApproving, isApprovingWrite, isRejectingWrite, onRequest, onApprove, onApproveWrite, onRejectWrite }: { agentName: string; gitEnabled: boolean; repo: RepositoryResource | null; workspace: GitWorkspace | null; writes: GitWrite[]; isRequesting: boolean; isApproving: boolean; isApprovingWrite: boolean; isRejectingWrite: boolean; onRequest: () => void; onApprove: (workspaceId: string) => void; onApproveWrite: (writeId: string) => void; onRejectWrite: (writeId: string) => void }) {
  const eligible = repo?.git_workspace?.eligible ?? false
  const blocked = repo && repo.git_workspace?.eligible === false
  const pendingWrites = writes.filter(item => item.status === 'pending')
  return (
    <div className="border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-cyan-400">Git workspace</div>
          <p className="mt-1 text-xs text-neutral-500">Isolated branch on the Mini. File writes, tests, and commits are approval-gated. Push is deferred.</p>
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
          <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words border border-neutral-800 bg-neutral-950 p-3 text-xs leading-relaxed text-neutral-400">{workspace.git_status}</pre>
          <div className="border border-neutral-800 bg-neutral-950/80 p-3">
            <div className="text-xs font-semibold text-cyan-400">Pending writes</div>
            {pendingWrites.length === 0 ? (
              <p className="mt-2 text-xs text-neutral-500">No pending write, test, or commit proposals.</p>
            ) : (
              <div className="mt-3 space-y-3">
                {pendingWrites.map(write => (
                  <div key={write.id} className="border border-neutral-800 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-xs text-neutral-200">{write.title || write.op}</div>
                        <div className="mt-1 text-xs text-neutral-400">{write.op}</div>
                      </div>
                      <WorkspaceStatus status="pending" />
                    </div>
                    <button disabled={isApprovingWrite} onClick={() => onApproveWrite(write.id)} className="mt-3 w-full bg-cyan-400 px-3 py-2 text-sm font-medium text-neutral-950 disabled:opacity-40">{isApprovingWrite ? 'Applying…' : 'Approve'}</button>
                    <button disabled={isRejectingWrite} onClick={() => onRejectWrite(write.id)} className="mt-2 w-full px-3 py-1.5 text-xs text-neutral-600 hover:text-red-300 disabled:opacity-30">Reject</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function WorkspaceMeta({ label, value }: { label: string; value: string }) {
  return <div><div className="text-xs text-neutral-400">{label}</div><div className="mt-1 break-all font-mono text-xs text-neutral-300">{value}</div></div>
}

function WorkspaceStatus({ status }: { status: GitWorkspace['status'] }) {
  const color = status === 'ready' ? 'border-green-800 text-green-300' : status === 'failed' ? 'border-red-900 text-red-300' : 'border-amber-700 text-amber-300'
  return <span className={`shrink-0 border px-2 py-1 text-xs ${color}`}>{status}</span>
}

function HandoffForm({ agents, assignments, draft, pending, error, onChange, onSubmit }: { agents: Agent[]; assignments: Assignment[]; draft: HandoffDefinition; pending: boolean; error: Error | null; onChange: (draft: HandoffDefinition) => void; onSubmit: () => void }) {
  const source = assignments.find(item => item.id === draft.source_assignment_id)
  return (
    <Panel title="New handoff" eyebrow="Handoff controls">
      <Field label="Completed assignment">
        <select value={draft.source_assignment_id} onChange={event => onChange({ ...draft, source_assignment_id: event.target.value, to_agent_id: '' })}>
          <option value="">Choose completed work</option>
          {assignments.map(assignment => <option key={assignment.id} value={assignment.id}>{assignment.title}</option>)}
        </select>
      </Field>
      <Field label="Target agent">
        <AgentPicker agents={agents} value={draft.to_agent_id} onChange={to_agent_id => onChange({ ...draft, to_agent_id })} placeholder="Choose the next specialist" excludeIds={source?.agent_id ? [source.agent_id] : []} />
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
      <div className="grid grid-cols-2 gap-3"><Meta label="Artifact" value={handoff.artifact_type} /><Meta label="Status" value={workStatus(handoff.status)} /></div>
      <TextBlock label="Instructions" value={handoff.instructions} />
      <TextBlock label="Artifact snapshot" value={handoff.artifact} muted />
      {assignment?.result && <TextBlock label="Downstream work product" value={assignment.result} />}
      {assignment?.error && <div className="border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">{assignment.error}</div>}
      {assignment && <OutputRecovery key={`recovery-${assignment.id}`} assignment={assignment} busy={isRunning} />}
      {assignment && <AssignmentHistory key={`history-${assignment.id}`} assignmentId={assignment.id} />}
      {error && <ErrorNotice error={error} />}
      {runnable && <button disabled={isRunning} onClick={onRun} className="w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{isRunning ? 'Mini is working…' : assignment?.status === 'failed' ? 'Retry downstream work' : 'Run downstream assignment'}</button>}
      <button disabled={isRemoving || handoff.status === 'running'} onClick={onRemove} className="w-full px-3 py-2 text-xs text-neutral-600 hover:text-red-300 disabled:opacity-30">Unlink handoff</button>
    </Panel>
  )
}

function Panel({ eyebrow, title, actions, children }: { eyebrow: string; title: string; actions?: React.ReactNode; children: React.ReactNode }) {
  return <><div className="flex items-start justify-between gap-3 border-b border-neutral-900 px-5 py-5 sm:px-6"><div className="min-w-0"><div className="text-xs font-semibold text-neutral-400">{eyebrow}</div><div role="heading" aria-level={2} className="mt-2 break-words text-lg text-white">{title}</div></div>{actions}</div><div className="space-y-5 px-5 py-6 sm:px-6">{children}</div></>
}

function Disclosure({ title, initiallyOpen = false, children }: { title: string; initiallyOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(initiallyOpen)
  return <details open={open} onToggle={event => setOpen(event.currentTarget.open)} className="group border-t border-neutral-800">
    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-3 text-sm text-neutral-300 hover:text-white [&::-webkit-details-marker]:hidden">
      {title}<ChevronDown size={16} aria-hidden="true" className="shrink-0 transition-transform group-open:rotate-180" />
    </summary>
    <div className="space-y-4 pb-3">{children}</div>
  </details>
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="agent-field block [&_input]:min-h-11 [&_select]:min-h-11"><span className="mb-2 block text-xs font-semibold text-neutral-400">{label}</span>{children}</label>
}

function Meta({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0 border border-neutral-800 bg-neutral-900/60 p-3"><div className="text-xs text-neutral-400">{label}</div><div className="mt-1 break-words text-sm text-neutral-200">{value}</div></div>
}

function TextBlock({ label, value, muted = false }: { label: string; value: string; muted?: boolean }) {
  return <section aria-label={label}><h3 className="mb-2 text-xs font-semibold text-neutral-400">{label}</h3><div tabIndex={0} className={`max-h-96 overflow-y-auto whitespace-pre-wrap break-words border border-neutral-800 bg-neutral-900/60 p-3 text-sm leading-relaxed ${muted ? 'text-neutral-400' : 'text-neutral-200'}`}>{value}</div></section>
}

function StatusLabel({ status }: { status: AssignmentStatus }) {
  const color = status === 'completed' ? 'border-green-800 text-green-300' : status === 'running' ? 'border-amber-700 text-amber-300' : status === 'failed' ? 'border-red-900 text-red-300' : 'border-neutral-700 text-neutral-400'
  return <span className={`max-w-full border px-2 py-1 text-xs ${color}`}>{workStatus(status)}</span>
}

function workStatus(status: AssignmentStatus) {
  return status === 'queued' ? 'Awaiting manual start' : status === 'running' ? 'Running' : status === 'failed' ? 'Failed' : 'Completed'
}


function ErrorNotice({ error }: { error: Error }) {
  return <div role="alert" className="border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">{error.message}</div>
}

function revisionError(message: string) {
  if (message.includes('review_changed_refresh_required')) return 'The review changed. Reload the assignment before requesting a revision.'
  if (message.includes('revision_persistence_failed')) return 'The revision was not queued. Retry when local storage is available.'
  if (message.includes('assignment_context_too_long')) return 'The original artifact is too large to carry into a revision.'
  if (message.includes('agent_not_found')) return 'The assigned agent no longer exists. Choose a new owner before revising.'
  return message
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
