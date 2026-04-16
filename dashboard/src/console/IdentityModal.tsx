import type { TeamUser } from '../lib/chatStore'

export function IdentityModal({
  onSelect,
  teamUsers,
}: {
  onSelect: (u: TeamUser) => void
  teamUsers: readonly TeamUser[]
}) {
  return (
    <div className="h-screen bg-[#0a0a0a] flex items-center justify-center">
      <div className="text-center max-w-sm">
        <div className="text-xs font-bold text-purple-400 tracking-[0.4em] mb-4">AIIA</div>
        <h1 className="text-3xl font-light text-white mb-2">Who are you?</h1>
        <p className="text-sm text-[#666] mb-8">
          So I know who I'm working with.
        </p>
        <div className="flex gap-3 justify-center">
          {teamUsers.map(u => (
            <button
              key={u}
              onClick={() => onSelect(u)}
              className="px-8 py-3 rounded-full border border-[#2a2a2a] text-[#ddd] hover:border-purple-500 hover:bg-purple-500/10 hover:text-white transition-all cursor-pointer"
            >
              {u}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
