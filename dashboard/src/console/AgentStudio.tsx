import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Agent, type AgentDefinition } from '../lib/api'
import { StudioTabs, type StudioView } from './StudioTabs'
import { WorkBoard } from './WorkBoard'

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
}

const EMPTY_AGENTS: Agent[] = []

const SKILL_LIBRARY = ['Research', 'Planning', 'Writing', 'Analysis', 'Coding', 'Memory']
const TOOL_LIBRARY = ['Local memory', 'Repository read', 'GitHub read', 'Git workspace']

export function AgentStudio() {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({ queryKey: ['agents'], queryFn: api.agents, refetchInterval: 10_000 })
  const { data: resources } = useQuery({ queryKey: ['agent-resources'], queryFn: api.agentResources })
  const agents = data?.agents ?? EMPTY_AGENTS
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [task, setTask] = useState('')
  const [view, setView] = useState<StudioView>('agents')
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
        }
      : EMPTY_DRAFT)
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

  if (view !== 'agents') {
    return <WorkBoard agents={agents} view={view} onViewChange={setView} />
  }

  return (
    <main className="min-h-0 flex flex-1 flex-col overflow-y-auto bg-neutral-950 lg:grid lg:grid-cols-[minmax(0,1fr)_360px] lg:overflow-hidden">
      <section className="relative min-w-0 shrink-0 border-b border-neutral-900 lg:overflow-hidden lg:border-r lg:border-b-0">
        <div className="flex flex-col gap-5 border-b border-neutral-900 px-5 py-6 sm:flex-row sm:items-start sm:justify-between sm:px-7">
          <div>
            <div className="text-[10px] font-semibold tracking-[0.28em] uppercase text-cyan-400">Agent Studio</div>
            <h1 className="mt-2 text-2xl font-medium text-white">Build a local team that can cook.</h1>
            <p className="mt-2 text-sm text-neutral-500">Define the role. Give it a task. The Mini runs it locally.</p>
          </div>
          <div className="flex shrink-0 flex-col items-start gap-3 sm:items-end">
            <StudioTabs view={view} onChange={setView} />
            <div className="hidden items-center gap-3 text-xs text-neutral-500 sm:flex">
              <span className="inline-flex items-center gap-2"><i className="h-2 w-2 rounded-full bg-green-500" />Mini online</span>
              <span>{activeCount} running</span>
            </div>
          </div>
        </div>

        <div className="relative min-h-[540px] overflow-y-auto px-5 py-8 sm:px-7 lg:h-[calc(100%-126px)]">
          <div className="absolute left-[50%] top-24 bottom-16 w-px bg-cyan-500/20" />
          <div className="relative mx-auto flex w-full max-w-4xl flex-col items-center gap-8">
            <div className="z-10 w-44 border border-cyan-400/50 bg-cyan-500/10 px-4 py-4 text-center shadow-[0_0_36px_rgba(34,211,238,0.08)]">
              <div className="text-[10px] tracking-[0.22em] uppercase text-cyan-300">Compute node</div>
              <div className="mt-1 text-base font-medium text-white">AIIA Mini</div>
              <div className="mt-1 text-[11px] text-neutral-500">qwen3:8b · local memory</div>
            </div>

            <div className="relative z-10 grid w-full grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {agents.map(agent => (
                <button
                  key={agent.id}
                  onClick={() => selectAgent(agent)}
                  className={`group min-h-44 border p-5 text-left transition-colors ${selectedId === agent.id ? 'border-cyan-400/70 bg-cyan-500/10' : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-600'}`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-base font-medium text-white">{agent.name}</div>
                      <div className="mt-1 line-clamp-2 text-xs leading-relaxed text-neutral-500">{agent.mission}</div>
                    </div>
                    <Status status={agent.status} />
                  </div>
                  <div className="mt-5 flex flex-wrap gap-1.5">
                    {agent.skills.slice(0, 4).map(skill => <span key={skill} className="border border-neutral-700 px-2 py-1 text-[10px] text-neutral-400">{skill}</span>)}
                  </div>
                  <div className="mt-5 flex items-center justify-between text-[10px] uppercase tracking-[0.16em] text-neutral-600"><span>{agent.last_run_at ? 'ran locally' : 'ready to run'}</span><span>{agent.loop_enabled ? `${agent.loop_interval_minutes}m loop` : 'manual'}</span></div>
                </button>
              ))}
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
      </section>

      <aside className="shrink-0 bg-neutral-950 lg:min-h-0 lg:overflow-y-auto">
        <div className="border-b border-neutral-900 px-6 py-5">
          <div className="text-[10px] font-semibold tracking-[0.24em] uppercase text-neutral-500">{selected ? 'Agent controls' : 'New agent'}</div>
          <div className="mt-2 text-lg text-white">{selected?.name || 'Define a role'}</div>
        </div>
        <div className="space-y-5 px-6 py-6">
          <Field label="Name"><input value={draft.name} onChange={event => setDraft({ ...draft, name: event.target.value })} placeholder="Signal Scout" /></Field>
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
          {selected && <button disabled={remove.isPending} onClick={() => remove.mutate()} className="w-full px-3 py-2 text-xs text-neutral-600 hover:text-red-300">Remove agent</button>}
        </div>

        {selected && <div className="border-t border-neutral-900 px-6 py-6">
          <div className="text-[10px] font-semibold tracking-[0.16em] uppercase text-cyan-400">Run on the Mini</div>
          <textarea value={task} onChange={event => setTask(event.target.value)} placeholder="Give this agent a focused task…" rows={4} className="mt-3 w-full border border-neutral-800 bg-neutral-900 px-3 py-2.5 text-sm text-white outline-none placeholder:text-neutral-700 focus:border-cyan-500/60" />
          <button disabled={!task.trim() || run.isPending || selected.status === 'running'} onClick={() => run.mutate()} className="mt-3 w-full bg-white px-3 py-2.5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40">{run.isPending || selected.status === 'running' ? 'Mini is working…' : 'Run agent'}</button>
          {(selected.last_result || selected.last_error) && <div className="mt-5 border border-neutral-800 bg-neutral-900/70 p-3"><div className="text-[10px] uppercase tracking-[0.14em] text-neutral-600">Latest run</div><p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-neutral-300">{selected.last_error || selected.last_result}</p></div>}
        </div>}
      </aside>
    </main>
  )
}

function Status({ status }: { status: Agent['status'] }) {
  const color = status === 'running' ? 'bg-amber-400' : status === 'error' ? 'bg-red-500' : 'bg-green-500'
  return <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${color}`} />
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="agent-field block"><span className="mb-2 block text-[10px] font-semibold tracking-[0.16em] uppercase text-neutral-500">{label}</span>{children}</label>
}
