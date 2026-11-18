import { studioFetch, studioJSON } from './connection'

export interface InsightSource {
  id: string
  job_id: string
  index: number
  bot_id: string
  created_at: string
  source_url: string
  source_quote: string
  signals: string[]
  insights: string[]
  review_status: string
  candidate_status: string
  provenance: { published_at?: string; captured_at?: string; processed_at?: string; publisher?: string; source_kind?: string; verification?: string; source_hash?: string }
}

export interface InsightProspect {
  id: string
  organization: string
  segment: string
  tenant_id: string
  packet_count: number
  pending_packet_count: number
  latest_at: string
  review_status: string
  source_url: string
  source_quote: string
  signals: string[]
  insights: string[]
  insight_status: string
  job_id: string
  index: number
  bot_id: string
  href: string
  sources: InsightSource[]
}

export interface BusinessInsights {
  as_of: string
  segment: string
  available_segments: { id: string; label: string }[]
  totals: { completed_jobs: number; failed_jobs: number; packets: number; prospects: number; insights: number; sources: number; awaiting_review: number }
  segments: { id: string; label: string; packets: number; prospects: number; insights: number }[]
  review: { status: string; count: number }[]
  evidence: { supported: number; inconclusive: number; rejected: number; unknown: number }
  job_outcomes: { supported: number; inconclusive: number; rejected: number; unknown: number }
  source_coverage: { with_quote: number; with_url: number; domains: number }
  trend: { date: string; packets: number; prospects: number }[]
  excluded_replay_jobs: number
  prospects: InsightProspect[]
}

export interface DemographicBand { label: string; value: number | null; share: number | null }
export interface Demographics {
  status: 'ok' | 'stale' | 'unavailable'
  geography: { id: string; name: string }
  geographies: { id: string; name: string }[]
  year: number
  period: string
  source: { name: string; url: string; api_url: string }
  fetched_at: string | null
  metrics: { population: number | null; households: number | null; median_household_income: number | null; median_age: number | null }
  age_bands: DemographicBand[]
  income_bands: DemographicBand[]
  error: string | null
}

export const consumerInsightsApi = {
  research: (segment = 'all') => studioFetch(`/api/bd/insights?segment=${encodeURIComponent(segment)}`).then(studioJSON<BusinessInsights>),
  demographics: (state = 'us') => studioFetch(`/api/consumer-insights/demographics?state=${encodeURIComponent(state)}`).then(studioJSON<Demographics>),
}
