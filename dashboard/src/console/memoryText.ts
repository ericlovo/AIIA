const MENTION = /<@[A-Z0-9]+>/g

/** The captured idea without the leading bot mention; the stored text is unchanged. */
export function captureText(text: string): string {
  return text.replace(MENTION, '').replace(/\s+/g, ' ').trim()
}

export type ReceiptTone = 'sent' | 'pending' | 'failed' | 'none'

/** Human label for a Slack receipt column pair; null status means no thread to notify. */
export function receiptLabel(status: string | null, error: string | null, noun: string): { tone: ReceiptTone; text: string } {
  if (status === 'sent') return { tone: 'sent', text: `${noun} receipt sent to Slack` }
  if (status === 'failed') return { tone: 'failed', text: `${noun} receipt failed${error ? `: ${error}` : ''}` }
  if (status === 'pending' || status === 'sending') return { tone: 'pending', text: `${noun} receipt queued for Slack` }
  return { tone: 'none', text: `No Slack thread for ${noun.toLowerCase()} receipt` }
}

export const MEMORY_POST_CHANNEL = '#aiia-memory'

/**
 * Label for an approved memory post; null when no post was requested for the capture.
 * A requested post with no outbox row means the Brain matched an existing memory that
 * another capture already queued: posts are keyed by memory, so it is not sent twice.
 */
export function memoryPostLabel(status: string | null, error: string | null, requested = false): { tone: ReceiptTone; text: string } | null {
  if (status === 'sent') return { tone: 'sent', text: `Posted to ${MEMORY_POST_CHANNEL}` }
  if (status === 'failed') return { tone: 'failed', text: `Post to ${MEMORY_POST_CHANNEL} failed${error ? `: ${error}` : ''}` }
  if (status === 'pending' || status === 'sending') return { tone: 'pending', text: `Post queued for ${MEMORY_POST_CHANNEL}` }
  if (requested) return { tone: 'none', text: `Same memory already posted to ${MEMORY_POST_CHANNEL} from another capture` }
  return null
}

export type PriorityTone = 'urgent' | 'high' | 'normal' | 'low'

/** Badge text for a stored priority; anything unrecognised reads as normal, matching the column default. */
export function priorityLabel(priority: string | null | undefined): { tone: PriorityTone; text: string } {
  const tone: PriorityTone = priority === 'urgent' || priority === 'high' || priority === 'low' ? priority : 'normal'
  return { tone, text: tone.charAt(0).toUpperCase() + tone.slice(1) }
}
