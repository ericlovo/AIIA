import { useEffect, useMemo, useState } from 'react'
import { loadChatState, saveChatState, TEAM_USERS, type TeamUser } from '../lib/chatStore'
import { TopBar } from './TopBar'
import { Mind } from './Mind'
import { Files } from './Files'
import { RightNow } from './RightNow'
import { Direct } from './Direct'
import { Pulse } from './Pulse'
import { IdentityModal } from './IdentityModal'
import { PanelBoundary } from './ErrorBoundary'

type LeftTab = 'mind' | 'files'

export function Console() {
  const initial = useMemo(() => loadChatState(), [])
  const [user, setUser] = useState<TeamUser | null>(initial.user)
  const [leftTab, setLeftTab] = useState<LeftTab>('mind')

  useEffect(() => {
    saveChatState({ ...loadChatState(), user })
  }, [user])

  if (!user) {
    return <IdentityModal onSelect={setUser} teamUsers={TEAM_USERS} />
  }

  return (
    <div className="h-screen flex flex-col bg-neutral-950 text-neutral-300 overflow-hidden">
      <PanelBoundary name="top bar">
        <TopBar user={user} onSwitchUser={() => setUser(null)} />
      </PanelBoundary>

      <div className="flex-1 min-h-0 grid grid-cols-[320px_1fr_440px] gap-px bg-neutral-900">
        {/* Left panel — Mind / Files tabs */}
        <div className="flex flex-col min-h-0">
          <div className="flex border-b border-neutral-900 bg-neutral-950 shrink-0">
            {(['mind', 'files'] as LeftTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setLeftTab(tab)}
                className={`flex-1 py-2 text-[10px] font-semibold tracking-widest uppercase cursor-pointer transition-colors ${
                  leftTab === tab
                    ? 'text-purple-400 border-b border-purple-500'
                    : 'text-neutral-600 hover:text-neutral-400'
                }`}
              >
                {tab === 'mind' ? 'Mind' : 'Files'}
              </button>
            ))}
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            {leftTab === 'mind' ? (
              <PanelBoundary name="mind"><Mind /></PanelBoundary>
            ) : (
              <PanelBoundary name="files"><Files /></PanelBoundary>
            )}
          </div>
        </div>

        <PanelBoundary name="right now"><RightNow user={user} /></PanelBoundary>
        <PanelBoundary name="direct"><Direct user={user} /></PanelBoundary>
      </div>

      <PanelBoundary name="pulse">
        <Pulse />
      </PanelBoundary>
    </div>
  )
}
