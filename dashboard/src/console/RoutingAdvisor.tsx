import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowRight, GitBranch } from 'lucide-react'
import { api, type Agent } from '../lib/api'

export function RoutingAdvisor({ agents, onSelect }: { agents: Agent[]; onSelect: (id: string) => void }) {
  const [brief, setBrief] = useState('')
  const [consent, setConsent] = useState(false)
  const status = useQuery({ queryKey: ['typesafe-status'], queryFn: api.typesafeStatus, retry: false })
  const fingerprint = JSON.stringify({ brief, candidates: agents.map(a => [a.id, a.name, a.skills]) })
  const advice = useMutation({
    mutationFn: async (input: { fingerprint: string; brief: string; ids: string[] }) => ({
      fingerprint: input.fingerprint,
      result: await api.suggestAssignmentAgent({ brief: input.brief, candidate_agent_ids: input.ids, allow_external: true }),
    }),
    retry: false,
  })
  const result = consent && advice.data?.fingerprint === fingerprint ? advice.data.result : null
  const selected = result && agents.find(agent => agent.id === result.agent_id)
  return <section aria-label="Jev routing advisor" className="space-y-3 border-t border-neutral-800 pt-3 text-sm">
    <h3 className="flex items-center gap-2 text-neutral-200"><GitBranch size={15} aria-hidden="true" />Jev routing advisor</h3>
    {!status.data?.ready ? <p className="text-neutral-400">{status.isLoading ? 'Checking connection...' : 'Not connected. Manual assignment is available.'}</p> : <>
      <label className="block text-neutral-300">Routing brief<textarea aria-label="Routing brief" maxLength={2000} rows={3} value={brief} onChange={event => setBrief(event.target.value)} className="mt-1 w-full border border-neutral-700 bg-neutral-950 p-2 text-white" /></label>
      <label className="flex items-start gap-2 text-xs text-neutral-400"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} />Allow this brief and candidate agent names and skills to leave the Mini for TypeSafe.</label>
      {agents.length > 32 && <p role="status">Routing advice supports up to 32 candidates. Choose an agent manually.</p>}
      <button type="button" disabled={!consent || !brief.trim() || !agents.length || agents.length > 32 || advice.isPending} onClick={() => advice.mutate({ brief, fingerprint, ids: agents.map(agent => agent.id) })} className="flex items-center gap-2 border border-neutral-700 px-3 py-2 text-white disabled:opacity-40"><GitBranch size={14} />{advice.isPending ? 'Requesting advice...' : 'Suggest specialist'}</button>
      {advice.isError && <p role="alert" className="text-amber-300">Routing advice unavailable. Choose an agent manually or try again later.</p>}
      {result && <div role="status" className="space-y-2 text-neutral-300">
        <p>{selected ? `Suggested: ${selected.name}` : 'No suitable specialist identified.'}</p>
        <p className="text-xs text-neutral-400">Confidence {(result.confidence * 100).toFixed(0)}% · {result.usage.input_tokens + result.usage.output_tokens} tokens · Cloud cost not estimated</p>
        {selected && <button type="button" onClick={() => onSelect(selected.id)} className="flex items-center gap-2 border border-cyan-700 px-3 py-2 text-cyan-200"><ArrowRight size={14} />Use suggested agent</button>}
      </div>}
    </>}
  </section>
}
