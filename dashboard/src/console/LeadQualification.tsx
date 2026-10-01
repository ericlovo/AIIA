import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type LeadReview, type LeadReviewDraft } from '../lib/api'

const EMPTY: LeadReviewDraft = { status: 'research', company: '', evidence_url: '', account_fit: '', observed_change: '', note: '' }

export function LeadQualification({ ideaId }: { ideaId: string }) {
  const [open, setOpen] = useState(false)
  const review = useQuery({ queryKey: ['lead-review', ideaId], queryFn: () => api.leadReview(ideaId), enabled: open, retry: false, staleTime: Infinity, refetchOnWindowFocus: false, refetchOnReconnect: false })
  return <details className="mt-4 border-t border-neutral-800 pt-3" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="min-h-11 cursor-pointer text-sm text-cyan-200">Lead qualification</summary>
    {open && review.isPending && <p role="status">Loading qualification...</p>}
    {open && review.isError && <p role="alert">Qualification unavailable. <button className="underline" onClick={() => void review.refetch()}>Retry</button></p>}
    {open && review.data && <ReviewForm key={`${ideaId}:${review.data.review?.version ?? 0}`} ideaId={ideaId} review={review.data.review} history={review.data.history} />}
  </details>
}

function ReviewForm({ ideaId, review, history }: { ideaId: string; review: LeadReview | null; history: LeadReview[] }) {
  const [draft, setDraft] = useState<LeadReviewDraft>(review ? { status: review.status, company: review.company, evidence_url: review.evidence_url, account_fit: review.account_fit, observed_change: review.observed_change, note: review.note } : EMPTY)
  const client = useQueryClient()
  const save = useMutation({
    mutationFn: () => api.saveLeadReview(ideaId, { ...draft, expected_version: review?.version ?? 0 }),
    onSuccess: data => client.setQueryData(['lead-review', ideaId], { review: data.review, history: [data.review, ...history].slice(0, 20) }),
  })
  const qualified = draft.status === 'qualified'
  const conflict = save.error?.message === 'qualification_changed_reload_before_saving'
  return <form className="max-w-3xl space-y-3 py-3 text-sm" onSubmit={event => { event.preventDefault(); save.mutate() }}>
    <p className="text-xs text-neutral-400">Human review only. Qualification does not authorize outreach or close the inbox item.</p>
    {review && <p role="status" className="text-xs text-emerald-300">Saved revision {review.version} · {new Date(review.updated_at).toLocaleString()}</p>}
    <label className="block">Decision<select aria-label="Decision" className="mt-1 block min-h-11 w-full border border-neutral-700 bg-neutral-900 px-3" value={draft.status} disabled={save.isPending} onChange={event => setDraft({ ...draft, status: event.target.value as LeadReviewDraft['status'] })}>
      <option value="research">Needs research</option><option value="watch">Watch</option><option value="qualified">Qualified for follow-up</option><option value="rejected">Not a fit</option>
    </select></label>
    {(['company', 'evidence_url', 'account_fit', 'observed_change', 'note'] as const).map(field => <label key={field} className="block">
      {{ company: 'Company', evidence_url: 'Primary-source URL (HTTPS)', account_fit: 'Account fit', observed_change: 'Observed change', note: 'Decision rationale' }[field]}
      <input className="mt-1 block min-h-11 w-full min-w-0 border border-neutral-700 bg-neutral-900 px-3" type={field === 'evidence_url' ? 'url' : 'text'} required={qualified || field === 'note'} maxLength={field === 'company' ? 200 : 2000} value={draft[field]} disabled={save.isPending} onChange={event => setDraft({ ...draft, [field]: event.target.value })} />
    </label>)}
    {save.isError && <p role="alert" className="text-red-300">{conflict ? 'Someone saved a newer review. Your draft has not been saved.' : 'Unable to save qualification. Check the fields and try again.'}</p>}
    {conflict && <button type="button" className="min-h-11 underline" onClick={() => void client.invalidateQueries({ queryKey: ['lead-review', ideaId] })}>Discard draft and load latest</button>}
    <button type="submit" disabled={save.isPending || conflict || !draft.note.trim()} className="min-h-11 border border-cyan-500 px-4 text-cyan-200 disabled:opacity-40">{save.isPending ? 'Saving...' : 'Save qualification'}</button>
    {history.length > 0 && <details><summary className="min-h-11 cursor-pointer text-neutral-400">Review history ({history.length})</summary><ol className="space-y-3">{history.map(item => <li key={item.version} className="break-words text-xs text-neutral-400">Revision {item.version} · {item.status} · {new Date(item.updated_at).toLocaleString()}<p className="mt-1 whitespace-pre-wrap">{item.note}</p></li>)}</ol></details>}
  </form>
}
