export interface DeliveryError { code: string; message: string }
export interface DeliveryAssignment {
  id: string; title?: string; status: string; agent_id?: string; agent_name?: string
  agent_status?: string; review_status?: string
}
export interface DeliveryRun {
  id: number; workflow_id?: number; name: string | null; html_url: string; head_sha: string
  status: string; conclusion: string | null; updated_at: string
}
export interface DeliveryPR {
  number: number; status: string; title?: string; state?: string; merged?: boolean; draft?: boolean; mergeable?: boolean | null
  head_sha?: string; html_url?: string; fetched_at?: string
  ci?: { status: string; scope: string; fetched_at?: string; head_sha?: string; runs: DeliveryRun[]; has_more?: boolean; errors: DeliveryError[] }
  errors: DeliveryError[]
}
export interface DeliveryFeature {
  id: string; title: string; status: string; priority: string; updated_at?: string; description?: string
  assignment_ids: string[]; pr_numbers: number[]; assignments: DeliveryAssignment[]; pull_requests: DeliveryPR[]
}
export interface ProjectDeliveryData {
  repo_id: string; checked_at: string; status: string; features: DeliveryFeature[]
  sources?: Record<string, { status: string; fetched_at?: string }>
  deployments?: { checked_at: string; status: string; notice?: string; errors: DeliveryError[]; environments: { environment: string; sha: string; state: string; updated_at: string; source_url: string; environment_url?: string | null }[] }
  assignment_candidates?: DeliveryAssignment[]; errors: DeliveryError[]
}
export function hasObservedWorkflowFailure(feature: DeliveryFeature): boolean {
  return feature.pull_requests.some(pr => {
    if (pr.ci?.status !== 'available' || !pr.head_sha) return false
    const latest = new Map<number, DeliveryRun>()
    for (const run of pr.ci.runs) {
      if (run.head_sha !== pr.head_sha || !run.workflow_id) continue
      const previous = latest.get(run.workflow_id)
      if (!previous || run.updated_at > previous.updated_at || (run.updated_at === previous.updated_at && run.id > previous.id)) latest.set(run.workflow_id, run)
    }
    return [...latest.values()].some(run => run.status === 'completed' && ['failure', 'timed_out', 'action_required', 'startup_failure'].includes(run.conclusion ?? ''))
  })
}
export function deliverySignals(feature: DeliveryFeature): string[] {
  const signals: string[] = []
  if (feature.assignments.some(a => a.status === 'running')) signals.push('Moving')
  if (feature.assignments.some(a => a.status === 'queued')) signals.push('Awaiting start')
  if (feature.assignments.some(a => a.status === 'missing')) signals.push('Broken link')
  if (feature.pull_requests.some(pr => pr.mergeable === false) || hasObservedWorkflowFailure(feature) || feature.status === 'blocked' || feature.assignments.some(a => ['failed', 'blocked'].includes(a.status) || a.review_status === 'rejected')) signals.push('Blocked')
  if (feature.assignments.some(a => a.status === 'completed' && a.review_status !== 'accepted')) signals.push('Needs review')
  if (!feature.assignment_ids.length && !feature.pr_numbers.length) signals.push('Unlinked')
  return signals.length ? signals : ['Linked']
}
export function parsePRNumbers(value: string): number[] {
  if (!value.trim()) return []
  const values = value.split(',').map(part => part.trim())
  if (values.some(part => !/^[1-9]\d*$/.test(part) || !Number.isSafeInteger(Number(part)))) throw new Error('Use positive PR numbers separated by commas.')
  return [...new Set(values.map(Number))]
}
