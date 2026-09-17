import type { SuiteGroup } from './graphLayout'

interface SuiteLegendProps {
  groups: SuiteGroup[]
  activeSuite: string | null
  onSelect: (slug: string | null) => void
  onTune: (slug: string) => void
}

export function SuiteLegend({ groups, activeSuite, onSelect, onTune }: SuiteLegendProps) {
  if (groups.length === 0) return null
  return (
    <div role="group" aria-label="Suite legend" className="flex min-w-0 max-w-full flex-wrap items-center gap-1.5">
      <span className="mr-1 text-[10px] font-semibold uppercase text-neutral-500">Suites</span>
      <button
        type="button"
        aria-pressed={activeSuite === null}
        onClick={() => onSelect(null)}
        className={`border px-2 py-1 ${activeSuite === null ? 'border-white/50 text-white' : 'border-neutral-800 text-neutral-400 hover:border-neutral-600'}`}
      >
        All agents
      </button>
      {groups.map(group => (
        <button
          key={group.slug}
          type="button"
          aria-pressed={activeSuite === group.slug}
          aria-label={`${group.slug} suite, ${group.count} ${group.count === 1 ? 'agent' : 'agents'}`}
          onClick={() => onSelect(activeSuite === group.slug ? null : group.slug)}
          className={`flex max-w-full items-center gap-1.5 border px-2 py-1 ${activeSuite === group.slug ? 'border-white/50 text-white' : 'border-neutral-800 text-neutral-400 hover:border-neutral-600'}`}
        >
          <i aria-hidden="true" className="h-2 w-2 shrink-0 rounded-full" style={{ background: group.color }} />
          <span className="truncate">{group.slug}</span>
          <span className="text-neutral-500">{group.count}</span>
        </button>
      ))}
      {activeSuite && (
        <button type="button" onClick={() => onTune(activeSuite)} className="border border-cyan-400/50 px-2 py-1 text-cyan-200 hover:border-cyan-200">
          Tune {activeSuite} suite
        </button>
      )}
    </div>
  )
}
