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

export type PriorityTone = 'urgent' | 'high' | 'normal' | 'low'

/** Badge text for a stored priority; anything unrecognised reads as normal, matching the column default. */
export function priorityLabel(priority: string | null | undefined): { tone: PriorityTone; text: string } {
  const tone: PriorityTone = priority === 'urgent' || priority === 'high' || priority === 'low' ? priority : 'normal'
  return { tone, text: tone.charAt(0).toUpperCase() + tone.slice(1) }
}
