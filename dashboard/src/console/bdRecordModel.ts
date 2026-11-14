import type { BDEvent, BDPacket } from './bdModel'

export interface BDReceipt {
  id: number
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
