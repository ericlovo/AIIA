import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Agent, type AgentDefinition, type AgentKind, type OutputChannel } from '../lib/api'
import { ONE_LINER_MAX, OUTPUT_CHANNELS } from './agentConfig'
import { ChannelChip } from './ChannelChip'
import { gitStanceLabel, groupAgentsByKind, pausedReviewLabel, productRepoLabel, resolveUseWhen } from './agentRoster'
import { filterAgents, hasSlackGap, noReviewedOutput, sortAgents, valueGlance, type ValueSort } from './agentValue'
import { AgentWorldCanvas } from './AgentWorldCanvas'
import { StudioNav } from './StudioNav'
import { VIEWS, type StudioView } from './studioRoute'
import { PageHeader } from './PageHeader'
import { navigate, replaceRoute, useStudioRoute } from './useStudioRoute'
import { WorkBoard } from './WorkBoard'
import { ActivityOverview } from './ActivityOverview'
import { Switchboard } from './Switchboard'
import { Today } from './Today'
import { Jobs } from './Jobs'
import { Projects } from './Projects'
import { MemoryLog } from './MemoryLog'
import { SignalJobs } from './SignalJobs'
import { AgentTimeline } from './AgentTimeline'
import { PanelBoundary } from './ErrorBoundary'
import type { ReviewBucket } from '../lib/api'

type Draft = AgentDefinition

const EMPTY_DRAFT: Draft = {
  name: '',
  mission: '',
  persona: 'Focused, pragmatic, and direct.',
  skills: [],
  tools: ['Local memory'],
  repo_id: '',
  temperature: 0.35,
  max_tokens: 1200,
  loop_enabled: false,
  loop_interval_minutes: 60,
  loop_task: '',
  loop_max_runs_per_day: 4,
  one_liner: '',
  output_channel: 'studio_inbox',
  kind: '',
  use_when: '',
  retired: false,
  handles: [],
}

const EMPTY_AGENTS: Agent[] = []

// Each view gets its own boundary, so one bad record cannot take the navigation with it.
const VIEW_NAMES = Object.fromEntries(VIEWS.map(item => [item.id, item.label])) as Record<StudioView, string>

const SKILL_LIBRARY = ['Research', 'Planning', 'Writing', 'Analysis', 'Coding', 'Memory']
const TOOL_LIBRARY = ['Local memory', 'Repository read', 'GitHub read', 'Git workspace']

export function AgentStudio() {
  const qc = useQueryClient()
  const { data, isLoading, isError } = useQuery({ queryKey: ['agents'], queryFn: api.agents, refetchInterval: 10_000 })
  const { data: resources } = useQuery({ queryKey: ['agent-resources'], queryFn: api.agentResources })
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000 })
  const { data: models } = useQuery({ queryKey: ['agent-models'], queryFn: api.agentModels, staleTime: 60_000, retry: false })
  const miniState = !health ? 'checking' : health.ollama?.status === 'online' ? 'online' : 'offline'
  const agents = data?.agents ?? EMPTY_AGENTS
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const inspector = useRef<HTMLElement>(null)
  useEffect(() => {
    if (selectedId && window.matchMedia('(max-width: 1023px)').matches) inspector.current?.scrollIntoView({ block: 'start' })
  }, [selectedId])
  const [inspectorView, setInspectorView] = useState<'activity' | 'configuration'>('activity')
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [task, setTask] = useState('')
  const [agentQuery, setAgentQuery] = useState('')
  const [onlyQuiet, setOnlyQuiet] = useState(false)
  const [showRetired, setShowRetired] = useState(false)
  const [agentSort, setAgentSort] = useState<ValueSort>('reviewed')
  const { route, key } = useStudioRoute()
  const view = route.view
  const routeAgentId = route.view === 'agents' ? route.agentId : undefined
  // Arriving at #/agents/:id selects that agent once, as soon as it has loaded.
  // Adjusting state during render (not in an effect) avoids a flash of the old selection.
  const [appliedKey, setAppliedKey] = useState('')
  if (view === 'agents' && appliedKey !== key) {
    const target = routeAgentId ? agents.find(agent => agent.id === routeAgentId) : undefined
    if (!routeAgentId) setAppliedKey(key)
    else if (target) { setAppliedKey(key); selectAgent(target) }
    else if (!isLoading) { setAppliedKey(key); selectAgent(null) }
  }
  // Once the arrival is applied, the address follows the selection on screen.
  useEffect(() => {
    if (view !== 'agents' || appliedKey !== key) return
    replaceRoute(selectedId ? { view: 'agents', agentId: selectedId } : { view: 'agents' })
  }, [view, appliedKey, key, selectedId])
  const selected = agents.find(agent => agent.id === selectedId) ?? null
  const needsRepo = draft.tools.some(tool => ['Repository read', 'GitHub read', 'Git workspace'].includes(tool))
  const githubConnected = resources?.github.status === 'connected'
  const canSave = draft.name.trim() && draft.mission.trim() && (!needsRepo || draft.repo_id)
  const selectedRepo = resources?.repos.find(repo => repo.id === draft.repo_id)

  function selectAgent(agent: Agent | null) {
    setSelectedId(agent?.id ?? null)
    setDraft(agent
      ? {
          name: agent.name, mission: agent.mission, persona: agent.persona, skills: agent.skills,
          tools: agent.tools, repo_id: agent.repo_id, temperature: agent.temperature,
          max_tokens: agent.max_tokens, loop_enabled: agent.loop_enabled,
          loop_interval_minutes: agent.loop_interval_minutes, loop_task: agent.loop_task,
          loop_max_runs_per_day: agent.loop_max_runs_per_day,
          model: agent.model ?? '', suite: agent.suite ?? '', memory_namespace: agent.memory_namespace ?? '',
          one_liner: agent.one_liner_derived ? '' : agent.one_liner ?? '',
          output_channel: agent.output_channel ?? 'studio_inbox',
          kind: agent.kind_derived ? '' : agent.kind ?? '',
          use_when: agent.use_when_derived ? '' : agent.use_when ?? '',
          retired: agent.retired ?? false,
          handles: agent.handles ?? [],
        }
      : EMPTY_DRAFT)
  }

  function changeView(nextView: StudioView) {
    navigate({ view: nextView })
  }

  function manageAgent(agentId: string) {
    navigate({ view: 'agents', agentId })
  }

  function assignAgent(agentId: string) {
    navigate({ view: 'assignments', agentId })
  }

  // A review metric is a doorway into the existing inbox, not a second screen.
  function openReview(bucket: ReviewBucket | '') {
    navigate({ view: 'memory', review: bucket || 'all' })
  }

  function openAssignment(assignmentId: string) {
    navigate({ view: 'assignments', assignmentId })
  }

  function routeHandoff(from: string, to: string) {
    navigate({ view: 'handoffs', from, to: to || undefined })
  }

  const save = useMutation({
    mutationFn: async () => {
      if (selected) return api.updateAgent(selected.id, draft)
      return api.createAgent(draft)
    },
    onSuccess: ({ agent }) => {
      selectAgent(agent)
      qc.invalidateQueries({ queryKey: ['agents'] })
    },
  })
  const run = useMutation({
    mutationFn: () => api.runAgent(selected!.id, task),
    onSuccess: ({ agent }) => {
      setTask('')
      setSelectedId(agent.id)
      qc.invalidateQueries({ queryKey: ['agents'] })
    },
  })
  const remove = useMutation({
    mutationFn: () => api.deleteAgent(selected!.id),
    onSuccess: () => {
      selectAgent(null)
      qc.invalidateQueries({ queryKey: ['agents'] })
    },
  })

  const activeCount = useMemo(() => agents.filter(agent => agent.status === 'running').length, [agents])
  const shownAgents = useMemo(() => {
    const filtered = sortAgents(filterAgents(agents, agentQuery, onlyQuiet), agentSort)
    return showRetired ? filtered : filtered.filter(agent => !agent.retired)
  }, [agents, agentQuery, onlyQuiet, agentSort, showRetired])
  const agentGroups = useMemo(() => groupAgentsByKind(shownAgents), [shownAgents])
  const quietCount = useMemo(() => agents.filter(noReviewedOutput).length, [agents])
  const retiredCount = useMemo(() => agents.filter(agent => agent.retired).length, [agents])

  const page = (() => {
    if (view === 'signals') return <SignalJobs />
    if (view === 'jobs') return <Jobs agents={agents} loading={isLoading} agentError={isError} />
    if (view === 'projects') return <Projects />
    if (view === 'switchboard' && !(route.view === 'switchboard' && route.taskId)) return <Today key={key} agents={agents} loading={isLoading} agentError={isError} attention={route.view === 'switchboard' && route.attention} />
    if (view === 'switchboard' || view === 'history') {
      return <Switchboard key={key} title={view === 'history' ? 'Activity history' : 'System task'} agents={agents} loading={isLoading} agentError={isError}
        onViewChange={changeView} onManageAgent={manageAgent} onAssignAgent={assignAgent}
        onOpenAssignment={openAssignment} onOpenReview={openReview} initialTaskId={route.view === 'switchboard' ? route.taskId : undefined}
        initialAttention={(route.view === 'switchboard' || route.view === 'history') && route.attention}
        onTemplate={template => { selectAgent(null); setDraft(template); changeView('agents') }} />
    }

    if (view === 'activity') {
      return <ActivityOverview agents={agents} isLoading={isLoading} />
    }

    if (view === 'world') {
      return (
        <AgentWorldCanvas
          agents={agents}
          loading={isLoading}
          agentError={isError}
          onManageAgent={manageAgent}
          onAssignAgent={assignAgent}
          onOpenAssignment={openAssignment}
          onRouteHandoff={routeHandoff}
        />
      )
    }

    if (view === 'inbox') return <MemoryLog key={key} agents={agents} inbox source={route.view === 'inbox' ? route.source ?? 'slack' : 'slack'} />
    if (view === 'memory') {
      const review = route.view === 'memory' ? route.review : undefined
      return <MemoryLog key={key} agents={agents} source={route.view === 'memory' ? route.source : undefined} intent={review ? { bucket: review === 'all' ? '' : review } : undefined} />
    }

    if (view !== 'agents') {
      return (
        <WorkBoard
          key={key}
          agents={agents}
          view={view}
          onRouteHandoff={routeHandoff}
          initialAgentId={route.view === 'assignments' ? route.agentId : undefined}
          initialAssignmentId={route.view === 'assignments' ? route.assignmentId : undefined}
          initialHandoffSourceId={route.view === 'handoffs' ? route.from : undefined}
          initialHandoffTargetId={route.view === 'handoffs' ? route.to : undefined}
        />
      )
    }

    return (
      <main className="min-h-0 flex h-full flex-1 flex-col overflow-y-auto bg-neutral-950 lg:grid lg:grid-cols-[minmax(0,1fr)_360px] lg:overflow-hidden">
        <section className="relative flex min-w-0 shrink-0 flex-col border-b border-neutral-900 lg:overflow-hidden lg:border-r lg:border-b-0">
          <PageHeader title="Agents" meta={<>
            <span>Define the role. Give it a task. The Mini runs it locally.</span>
            <span className="inline-flex items-center gap-2"><i aria-hidden="true" className={`h-2 w-2 rounded-full ${miniState === 'online' ? 'bg-green-500' : miniState === 'offline' ? 'bg-red-500' : 'bg-neutral-600'}`} />{miniState === 'checking' ? 'Checking Mini…' : `Mini ${miniState}`}</span>
            <span>{activeCount} running</span>
          </>} />

          <div className="relative min-h-[540px] overflow-y-auto px-5 py-8 sm:px-7 lg:min-h-0 lg:flex-1">
            <div className="absolute left-[50%] top-24 bottom-16 w-px bg-cyan-500/20" />
            <div className="relative mx-auto flex w-full max-w-4xl flex-col items-center gap-8">
              <div className="z-10 w-44 border border-cyan-400/50 bg-cyan-500/10 px-4 py-4 text-center shadow-[0_0_36px_rgba(34,211,238,0.08)]">
                <div className="text-[10px] tracking-[0.22em] uppercase text-cyan-300">Compute node</div>
                <div className="mt-1 text-base font-medium text-white">AIIA Mini</div>
                <div className="mt-1 text-[11px] text-neutral-500">{models?.default || 'Model unknown'} · local memory</div>
              </div>

              <div className="relative z-10 mb-4 flex w-full max-w-4xl flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                <label className="min-w-0 flex-1 text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">Find agent<input value={agentQuery} onChange={event => setAgentQuery(event.target.value)} placeholder="Name, use-when, or repo" className="mt-2 w-full border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm normal-case tracking-normal text-white outline-none focus:border-cyan-500/60" /></label>
                <label className="text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">Sort<select aria-label="Sort agents" value={agentSort} onChange={event => setAgentSort(event.target.value as ValueSort)} className="mt-2 border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm normal-case tracking-normal text-white outline-none focus:border-cyan-500/60"><option value="reviewed">Least reviewed first</option><option value="runs">Most runs</option><option value="last_run">Last run</option><option value="name">Name</option></select></label>
                <button type="button" aria-pressed={onlyQuiet} onClick={() => setOnlyQuiet(!onlyQuiet)} className={`mt-6 border px-3 py-2 text-xs ${onlyQuiet ? 'border-amber-400/70 bg-amber-500/10 text-amber-100' : 'border-neutral-800 text-neutral-400'}`}>No reviewed output in 14 days ({quietCount})</button>
                <button type="button" aria-pressed={showRetired} onClick={() => setShowRetired(!showRetired)} className={`mt-6 border px-3 py-2 text-xs ${showRetired ? 'border-cyan-400/70 bg-cyan-500/10 text-cyan-100' : 'border-neutral-800 text-neutral-400'}`}>Show retired ({retiredCount})</button>
              </div>
              <div className="relative z-10 flex w-full flex-col gap-8">
                {agentGroups.map(group => (
                  <section key={group.id || 'unsorted'} aria-label={group.label} className="flex flex-col gap-3">
                    <h2 className="text-[10px] font-semibold tracking-[0.22em] uppercase text-cyan-300/80">{group.label}</h2>
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                      {group.agents.map(agent => {
                        const paused = pausedReviewLabel(agent)
                        return (
                          <button
                            key={agent.id}
                            onClick={() => selectAgent(agent)}
                            className={`group min-h-44 border p-5 text-left transition-colors ${selectedId === agent.id ? 'border-cyan-400/70 bg-cyan-500/10' : noReviewedOutput(agent) ? 'border-amber-500/40 bg-amber-950/20 hover:border-amber-400/60' : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-600'}`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="truncate text-base font-medium text-white">{agent.name}</div>
                                <div className="mt-1 line-clamp-2 text-xs leading-relaxed text-neutral-400">{resolveUseWhen(agent) || agent.mission}</div>
                              </div>
                              <Status status={agent.status} />
                            </div>
                            <div className="mt-3 flex flex-wrap gap-1.5">
                              <span className="border border-neutral-700 px-2 py-1 text-[10px] text-neutral-300">{productRepoLabel(agent)}</span>
                              <span className={`border px-2 py-1 text-[10px] ${agent.tools.includes('Git workspace') ? 'border-cyan-500/50 text-cyan-200' : 'border-neutral-700 text-neutral-400'}`}>{gitStanceLabel(agent)}</span>
                              {paused && <span className="border border-amber-400/50 px-2 py-1 text-[10px] text-amber-200">{paused}</span>}
                              {agent.retired && <span className="border border-neutral-600 px-2 py-1 text-[10px] text-neutral-400">Retired</span>}
                            </div>
                            <div className="mt-3 flex flex-wrap gap-1.5">
                              <ChannelChip channel={agent.output_channel} note={agent.output_channel_note} />
                              <span className={`border px-2 py-1 text-[10px] ${noReviewedOutput(agent) ? 'border-amber-400/50 text-amber-200' : 'border-neutral-700 text-neutral-400'}`}>{valueGlance(agent)}</span>
                              {hasSlackGap(agent) && <span className="border border-amber-400/50 px-2 py-1 text-[10px] text-amber-200">{agent.output_channel_note}</span>}
                              {(agent.handles ?? []).slice(0, 4).map(tag => <span key={tag} className="border border-neutral-700 px-2 py-1 text-[10px] text-neutral-400">{tag}</span>)}
                            </div>
                            <div className="mt-5 flex items-center justify-between text-[10px] uppercase tracking-[0.16em] text-neutral-600"><span>{agent.last_run_at ? 'ran locally' : 'ready to run'}</span><span>{agent.loop_enabled ? `${agent.loop_interval_minutes}m loop` : 'manual'}</span></div>
                          </button>
                        )
                      })}
                    </div>
                  </section>
                ))}
                {shownAgents.length === 0 && agents.length > 0 && (
                  <p className="text-sm text-neutral-500">No agents match this filter.</p>
                )}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {!isLoading && agents.length === 0 && (
                    <button onClick={() => selectAgent(null)} className="min-h-44 border border-dashed border-cyan-500/40 bg-cyan-500/[0.03] p-5 text-left hover:bg-cyan-500/[0.07]">
                      <div className="text-sm font-medium text-cyan-300">Create the first agent</div>
                      <p className="mt-2 text-xs leading-relaxed text-neutral-500">Start with a researcher, operator, strategist, or domain expert.</p>
                    </button>
                  )}
                  <button onClick={() => selectAgent(null)} className="min-h-44 border border-dashed border-neutral-700 p-5 text-left text-neutral-500 hover:border-cyan-500/50 hover:text-cyan-300">
                    <div className="text-2xl font-light">+</div>
                    <div className="mt-4 text-sm">New local agent</div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <aside ref={inspector} className="shrink-0 bg-neutral-950 lg:min-h-0 lg:overflow-y-auto">
          {selected && <div className="flex gap-2 border-b border-neutral-800 p-3" aria-label="Agent view">
            {(['activity', 'configuration'] as const).map(value => <button key={value} aria-pressed={inspectorView === value} onClick={() => setInspectorView(value)} className={`min-h-11 flex-1 px-3 text-sm ${inspectorView === value ? 'bg-neutral-800 text-white' : 'text-neutral-400'}`}>{value === 'activity' ? 'Activity' : 'Configuration'}</button>)}
          </div>}
          {selected && inspectorView === 'activity' ? <AgentTimeline key={selected.id} agent={selected} /> : <>
          <div className="border-b border-neutral-900 px-6 py-5">
            <div className="text-[10px] font-semibold tracking-[0.24em] uppercase text-neutral-500">{selected ? 'Agent controls' : 'New agent'}</div>
            <div className="mt-2 text-lg text-white">{selected?.name || 'Define a role'}</div>
          </div>
          <div className="space-y-5 px-6 py-6">
            <Field label="Name"><input value={draft.name} onChange={event => setDraft({ ...draft, name: event.target.value })} placeholder="Signal Scout" /></Field>
            <Field label="One-liner"><input value={draft.one_liner ?? ''} maxLength={ONE_LINER_MAX} onChange={event => setDraft({ ...draft, one_liner: event.target.value })} placeholder="One sentence: what this agent is for." /></Field>
            <Field label="Use when"><input value={draft.use_when ?? ''} maxLength={ONE_LINER_MAX} onChange={event => setDraft({ ...draft, use_when: event.target.value })} placeholder="When should someone pick this agent?" /></Field>
            <Field label="Kind">
              <select aria-label="Kind" className="w-full border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60" value={draft.kind ?? ''} onChange={event => setDraft({ ...draft, kind: event.target.value as AgentKind | '' })}>
                <option value="">Derive from repo/tools</option>
                <option value="coding">Coding</option>
                <option value="product">Product</option>
                <option value="ops">Ops</option>
              </select>
            </Field>
            <label className="flex items-center gap-2 text-sm text-neutral-300"><input type="checkbox" checked={Boolean(draft.retired)} onChange={event => setDraft({ ...draft, retired: event.target.checked })} className="accent-cyan-400" />Retired</label>
            <Field label="Handles"><input value={(draft.handles ?? []).join(', ')} onChange={event => setDraft({ ...draft, handles: event.target.value.split(',').map(tag => tag.trim()).filter(Boolean).slice(0, 12) })} placeholder="ci, review, brief" /></Field>
            <Field label="Output channel">
              <select aria-label="Output channel" className="w-full border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60" value={draft.output_channel ?? 'studio_inbox'} onChange={event => setDraft({ ...draft, output_channel: event.target.value as OutputChannel })}>
                {OUTPUT_CHANNELS.map(channel => <option key={channel.id} value={channel.id}>{channel.label}</option>)}
              </select>
            </Field>
            {draft.output_channel === 'slack' && <p className="text-xs text-amber-200/90">Slack is a declared destination only. Nothing posts agent output to Slack yet, so results are delivered to the Studio inbox and the agent shows why.</p>}
            <Field label="Mission"><textarea value={draft.mission} onChange={event => setDraft({ ...draft, mission: event.target.value })} placeholder="Watch a domain, find signal, and make a clear recommendation." rows={3} /></Field>
            <Field label="Persona"><textarea value={draft.persona} onChange={event => setDraft({ ...draft, persona: event.target.value })} rows={3} /></Field>
            <div>
              <div className="mb-2 text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">Skills</div>
              <div className="flex flex-wrap gap-2">
                {SKILL_LIBRARY.map(skill => {
                  const selectedSkill = draft.skills.includes(skill)
                  return <button key={skill} onClick={() => setDraft({ ...draft, skills: selectedSkill ? draft.skills.filter(item => item !== skill) : [...draft.skills, skill] })} className={`border px-2.5 py-1.5 text-xs ${selectedSkill ? 'border-cyan-400/60 bg-cyan-500/10 text-cyan-200' : 'border-neutral-800 text-neutral-500 hover:border-neutral-600'}`}>{skill}</button>
                })}
              </div>
            </div>
            <div>
              <div className="mb-2 text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">Tools</div>
              <div className="flex flex-wrap gap-2">
                {TOOL_LIBRARY.map(tool => {
                  const selectedTool = draft.tools.includes(tool)
                  return <button key={tool} onClick={() => setDraft({ ...draft, tools: selectedTool ? draft.tools.filter(item => item !== tool) : [...draft.tools, tool] })} className={`border px-2.5 py-1.5 text-xs ${selectedTool ? 'border-cyan-400/60 bg-cyan-500/10 text-cyan-200' : 'border-neutral-800 text-neutral-500 hover:border-neutral-600'}`}>{tool}</button>
                })}
              </div>
              {needsRepo && <select aria-label="Repository" className="mt-3 w-full border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60" value={draft.repo_id} onChange={event => setDraft({ ...draft, repo_id: event.target.value })}>
                <option value="">Choose repository</option>
                {(resources?.repos ?? []).map(repo => <option key={repo.id} value={repo.id}>{repo.name}{repo.dirty ? ' · modified' : ''}</option>)}
              </select>}
              {needsRepo && selectedRepo && <p className="mt-2 break-words text-xs text-neutral-500">{selectedRepo.branch} · {selectedRepo.github_repo || 'local only'}</p>}
              {draft.tools.includes('GitHub read') && <p className={`mt-2 text-xs ${githubConnected ? 'text-emerald-300/80' : 'text-amber-300/80'}`}>{githubConnected ? `Connected · @${resources.github.account} · read only` : `GitHub ${resources?.github.status ?? 'checking'}`}</p>}
              {draft.tools.includes('Git workspace') && <p className={`mt-2 text-xs ${selectedRepo?.git_workspace?.eligible ? 'text-cyan-300/80' : 'text-amber-300/80'}`}>{selectedRepo?.git_workspace?.eligible ? 'Isolated worktrees · human approval required' : selectedRepo ? 'Git workspace blocked · verify repository remote' : 'Choose a repository for isolated worktrees'}</p>}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Temperature"><input type="number" min="0" max="1" step="0.05" value={draft.temperature} onChange={event => setDraft({ ...draft, temperature: Number(event.target.value) })} /></Field>
              <Field label="Max tokens"><input type="number" min="128" max="2000" step="128" value={draft.max_tokens} onChange={event => setDraft({ ...draft, max_tokens: Number(event.target.value) })} /></Field>
            </div>
            <div className="border border-neutral-800 bg-neutral-900/50 p-4">
              <div className="flex items-center justify-between gap-3"><div><div className="text-[10px] font-semibold tracking-[0.16em] uppercase text-cyan-400">Loop node</div><p className="mt-1 text-xs text-neutral-500">Run a bounded recurring task on the Mini.</p></div><button onClick={() => setDraft({ ...draft, loop_enabled: !draft.loop_enabled })} className={`border px-2.5 py-1.5 text-xs ${draft.loop_enabled ? 'border-cyan-400/60 text-cyan-200' : 'border-neutral-700 text-neutral-500'}`}>{draft.loop_enabled ? 'Enabled' : 'Disabled'}</button></div>
              {draft.loop_enabled && <div className="mt-4 space-y-3"><Field label="Loop task"><textarea value={draft.loop_task} onChange={event => setDraft({ ...draft, loop_task: event.target.value })} rows={3} placeholder="Inspect the mounted repository and report only material changes." /></Field><div className="grid grid-cols-2 gap-3"><Field label="Every minutes"><input type="number" min="15" max="1440" value={draft.loop_interval_minutes} onChange={event => setDraft({ ...draft, loop_interval_minutes: Number(event.target.value) })} /></Field><Field label="Runs / day"><input type="number" min="1" max="48" value={draft.loop_max_runs_per_day} onChange={event => setDraft({ ...draft, loop_max_runs_per_day: Number(event.target.value) })} /></Field></div></div>}
            </div>
            <button disabled={!canSave || save.isPending} onClick={() => save.mutate()} className="w-full bg-cyan-400 px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{save.isPending ? 'Saving…' : selected ? 'Save agent' : 'Create agent'}</button>
            {save.isError && <p role="alert" className="text-xs text-red-300">{save.error.message}</p>}
            {selected && <button disabled={remove.isPending} onClick={() => remove.mutate()} className="w-full px-3 py-2 text-xs text-neutral-600 hover:text-red-300">Remove agent</button>}
            {remove.isError && <p role="alert" className="text-xs text-red-300">{remove.error.message}</p>}
          </div>

          {selected && <div className="border-t border-neutral-900 px-6 py-6">
            <div className="text-[10px] font-semibold tracking-[0.16em] uppercase text-cyan-400">Run on the Mini</div>
            <textarea value={task} onChange={event => setTask(event.target.value)} placeholder="Give this agent a focused task…" rows={4} className="mt-3 w-full border border-neutral-800 bg-neutral-900 px-3 py-2.5 text-sm text-white outline-none placeholder:text-neutral-700 focus:border-cyan-500/60" />
            <button disabled={!task.trim() || run.isPending || selected.status === 'running'} onClick={() => run.mutate()} className="mt-3 w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{run.isPending || selected.status === 'running' ? 'Mini is working…' : 'Run agent'}</button>
            {run.isError && <p role="alert" className="mt-3 text-xs text-red-300">{run.error.message}</p>}
            {(selected.last_result || selected.last_error) && <div className="mt-5 border border-neutral-800 bg-neutral-900/70 p-3"><div className="text-[10px] uppercase tracking-[0.14em] text-neutral-600">Latest run</div><p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-neutral-300">{selected.last_error || selected.last_result}</p></div>}
          </div>}
          </>}
        </aside>
      </main>
    )
  })()

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative z-30 flex shrink-0 items-center border-b border-neutral-900 px-3 py-2 sm:px-7">
        <StudioNav view={view} />
      </div>
      <div className="min-h-0 flex-1">
        <PanelBoundary key={view} name={VIEW_NAMES[view]}>{page}</PanelBoundary>
      </div>
    </div>
  )
}

function Status({ status }: { status: Agent['status'] }) {
  const color = status === 'running' ? 'bg-amber-400' : status === 'error' ? 'bg-red-500' : 'bg-green-500'
  return <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${color}`} />
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="agent-field block"><span className="mb-2 block text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">{label}</span>{children}</label>
}

