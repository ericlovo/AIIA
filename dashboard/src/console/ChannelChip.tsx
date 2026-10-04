import type { OutputChannel } from '../lib/api'

export function ChannelChip({ channel, note }: { channel?: OutputChannel | string; note?: string }) {
  const slack = channel === 'slack'
  return <span className={`border px-2 py-1 text-[10px] uppercase tracking-[0.12em] ${slack ? 'border-sky-500/50 text-sky-200' : 'border-emerald-700/60 text-emerald-200'}`}>{slack ? 'Slack' : 'Inbox'}{note ? ' · fallback' : ''}</span>
}
