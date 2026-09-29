import { test } from 'node:test'
import assert from 'node:assert/strict'
import { bulkDismissError } from '../src/console/assignmentReview.ts'

test('a refused bulk dismissal says nothing was dismissed and names the blocking item', () => {
  const titles = (id: string) => (id === 'asg_1' ? 'Scheduled: CI Signal Officer' : '')
  assert.equal(bulkDismissError('review_changed_refresh_required:asg_1', titles),
    'Nothing was dismissed. "Scheduled: CI Signal Officer" changed after you selected it. Re-check it, then dismiss again.')
  assert.match(bulkDismissError('assignment_not_found:asg_gone', titles), /^Nothing was dismissed\. "asg_gone" no longer exists/)
  assert.equal(bulkDismissError('500 Internal Server Error', titles), 'Nothing was dismissed. 500 Internal Server Error')
})
