import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { decisionQueue } from './decisionQueue'

export function useDecisionQueue() {
  const assignments = useQuery({ queryKey: ['assignments', 'summary'], queryFn: api.assignmentSummaries, retry: false, refetchInterval: 5_000 })
  const workspaces = useQuery({ queryKey: ['git-workspaces'], queryFn: api.gitWorkspaces, retry: false, refetchInterval: 5_000 })
  const writes = useQuery({ queryKey: ['git-writes'], queryFn: () => api.gitWrites(), retry: false, refetchInterval: 5_000 })
  const ideas = useQuery({ queryKey: ['memory-inbox', 'decision-queue'], queryFn: () => api.memoryInbox({ status: 'unreviewed' }), retry: false, refetchInterval: 5_000 })
  const queries = [assignments, workspaces, writes, ideas]
  return {
    ...decisionQueue(assignments.data?.assignments ?? [], workspaces.data?.workspaces ?? [], writes.data?.writes ?? [], ideas.data?.attention_counts ? ideas.data.attention_counts.needs_you ?? 0 : ideas.data?.total ?? 0),
    verification: ideas.data?.attention_counts?.verification ?? 0,
    activity: ideas.data?.attention_counts?.activity ?? 0,
    loading: queries.some(query => query.isPending),
    error: queries.some(query => query.isError),
  }
}
