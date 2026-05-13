import { useEffect, useMemo, useState } from 'react'
import { loadChatState, saveChatState, TEAM_USERS, type TeamUser } from '../lib/chatStore'
import { TopBar } from './TopBar'
import { Mind } from './Mind'
import { RightNow } from './RightNow'
import { Direct } from './Direct'
import { Pulse } from './Pulse'
import { IdentityModal } from './IdentityModal'
import { PanelBoundary } from './ErrorBoundary'

export function Console() {
  const initial = useMemo(() => loadChatState(), [])
  const [user, setUser] = useState<TeamUser | null>(initial.user)

  // Persist identity every time it changes
  useEffect(() => {
    saveChatState({ ...loadChatState(), user })
  }, [user])

  if (!user) {
    return <IdentityModal onSelect={setUser} teamUsers={TEAM_USERS} />
  }

  return (
    <div className="h-screen flex flex-col bg-neutral-950 text-neutral-300 overflow-hidden">
      {/* Top: system vitals + identity */}
      <PanelBoundary name="top bar">
        <TopBar user={user} onSwitchUser={() => setUser(null)} />
      </PanelBoundary>

      {/* Three lanes — the heart of the console */}
      <div className="flex-1 min-h-0 grid grid-cols-[320px_1fr_440px] gap-px bg-neutral-900">
        <PanelBoundary name="mind"><Mind /></PanelBoundary>
        <PanelBoundary name="right now"><RightNow user={user} /></PanelBoundary>
        <PanelBoundary name="direct"><Direct user={user} /></PanelBoundary>
      </div>

      {/* Bottom: pulse strip showing all autonomous loops */}
      <PanelBoundary name="pulse">
        <Pulse />
      </PanelBoundary>
    </div>
  )
}
