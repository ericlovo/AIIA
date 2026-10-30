import type { Assignment, GitWorkspace, GitWrite } from '../lib/api'
import { groupWork, workLane } from './workQueue.ts'

export function decisionQueue(assignments: Assignment[], workspaces: GitWorkspace[], writes: GitWrite[], savedIdeas = 0) {
  const pendingApprovalIds = new Set([
    ...workspaces.filter(item => item.status === 'pending' || writes.some(write => write.workspace_id === item.id && write.status === 'pending')).map(item => item.assignment_id),
    ...writes.filter(item => item.status === 'pending').map(item => item.assignment_id),
  ])
  const needsYou = assignments.filter(item => workLane(item, pendingApprovalIds.has(item.id)) === 'needs-you')
  const missingApprovalIds = [...pendingApprovalIds].filter(id => !assignments.some(item => item.id === id))
  return { pendingApprovalIds, needsYou, groups: groupWork(needsYou).sort((a, b) => Number(b.some(item => pendingApprovalIds.has(item.id))) - Number(a.some(item => pendingApprovalIds.has(item.id)))), savedIdeas, missingApprovalIds, total: needsYou.length + savedIdeas + missingApprovalIds.length }
}
