import type { BDEvent, BDPacket } from './bdModel'

export type ReceiptId = number | string

// SQLite uses positive integers. PostgreSQL returns canonical UUID strings.
export function isReceiptId(value: unknown): value is ReceiptId {
  if (typeof value === 'number') return Number.isSafeInteger(value) && value > 0
  return typeof value === 'string' && value.length === 36 &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value)
}

export interface BDReceipt {
  id: ReceiptId
  job_id: string
  packet_index: number
  tenant_id: string
  bot_id: string
  organization: string
  mode: string
  status: string
  created_at: string
}

export interface BDReceiptDetail {
  receipt: BDReceipt
  packet: BDPacket
  stages: BDEvent[]
}

export function bdDecisionLabel(status: string) {
  switch (status) {
    case 'research_next': return 'Research next'
    case 'watch': return 'Watch'
    case 'dismissed': return 'Dismissed'
    case 'human_review_required': return 'Restored for review'
    default: return 'Unknown decision'
  }
}
