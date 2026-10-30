import { groupIdeaEvidence } from './ideaEvidence'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type IdeaLane } from '../lib/api'
import { formatRoute } from './studioRoute'

const LABELS: Record<string, string> = { needs_you: 'Needs you', verification: 'Needs verification', activity: 'Activity', linked_work: 'Linked work' }
export function IdeaAttention() {
  const [view, setView] = useState<IdeaLane | 'preview' | null>(null)
  const [offset, setOffset] = useState(0)
  const page = useQuery({ queryKey: ['memory-inbox', 'attention', view, offset], queryFn: () => api.memoryInbox({ status: 'unreviewed', attention: view as IdeaLane, offset }), enabled: Boolean(view && view !== 'preview'), retry: false })
  const preview = useQuery({ queryKey: ['memory-inbox', 'attention-preview', offset], queryFn: () => api.ideaAttentionPreview(offset), enabled: view === 'preview', retry: false })
  const active = view === 'preview' ? preview : page
  const data = active.data
  return <section aria-label="Saved idea routing" className="my-4 min-w-0 border border-neutral-700 p-4 text-sm text-neutral-300">
    <h2 className="font-semibold">Decisions and verification</h2>
    <p className="my-2">New automated findings need verification before a human decision. Digests are activity. Existing records keep their current placement until classification is approved.</p>
    <div className="flex flex-wrap gap-2">{(['needs_you', 'verification', 'activity', 'preview'] as const).map(lane => <button key={lane} aria-pressed={view === lane} onClick={() => { setView(view === lane ? null : lane); setOffset(0) }} className="min-h-11 border border-neutral-600 px-3 py-2">{lane === 'preview' ? 'Preview all classifications' : LABELS[lane]}</button>)}</div>
    {view && <div className="mt-4">
      {view === 'preview' && <p role="status">Preview only — no records changed. No classifications can be applied here.</p>}
      {active.isPending && <p role="status">Loading records…</p>}
      {active.isError && <p role="alert">Records unavailable. Refresh before deciding; this is not an empty queue.</p>}
      {!active.isError && data && <>
        <p className="my-3">{data.total} records · {data.total ? offset + 1 : 0}–{Math.min(offset + 50, data.total)}</p>
        {view === 'preview' && preview.data?.counts && <p>{Object.entries(preview.data.counts).map(([lane, count]) => `${LABELS[lane] ?? lane}: ${count}`).join(' · ')}</p>}
        <ul className="space-y-3">{groupIdeaEvidence(data.ideas ?? []).map(group => <li key={group[0].id} className="break-words border-t border-neutral-700 pt-3">
          {group.length > 1 && <p>{group.length} matching claims on this page · all original evidence retained</p>}
          {group.map(idea => <div key={idea.id}>
          <details><summary className="min-h-11 cursor-pointer py-2">{idea.text.split('\n')[0]} · {idea.source}</summary>
            {'proposed_lane' in idea && 'reason' in idea && <p className="my-2">Proposed: {LABELS[String(idea.proposed_lane)] ?? String(idea.proposed_lane)} — {String(idea.reason)}</p>}
            <p className="whitespace-pre-wrap">{idea.text}</p>
            <p className="my-2 break-all text-xs">Original {idea.id} · {idea.created_at} · {idea.status}</p>
            {idea.review_outcome && <p>Existing review: {idea.review_outcome} {idea.review_note}</p>}
            {idea.assignment_id && <a className="inline-flex min-h-11 items-center text-cyan-200 underline" href={formatRoute({ view: 'assignments', assignmentId: idea.assignment_id })}>Open linked decision and evidence</a>}
            <p>Viewing this evidence does not accept output, approve execution, or post a message.</p>
          </details></div>)}
        </li>)}</ul>
        <div className="mt-3 flex gap-3"><button className="min-h-11 border px-3 disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous records</button><button className="min-h-11 border px-3 disabled:opacity-40" disabled={offset + 50 >= data.total} onClick={() => setOffset(offset + 50)}>Next records</button></div>
      </>}
    </div>}
  </section>
}
