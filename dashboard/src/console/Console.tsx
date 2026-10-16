import { TopBar } from './TopBar'
import { Pulse } from './Pulse'
import { PanelBoundary } from './ErrorBoundary'
import { AgentStudio } from './AgentStudio'
import { VoiceConductor } from './VoiceConductor'
import { useStudioRoute } from './useStudioRoute'

export function Console() {
  const [voiceOpen, setVoiceOpen] = useState(false)
  const [statusOpen, setStatusOpen] = useState(false)
  const { route } = useStudioRoute()
  const home = route.view === 'home'
  return (
    <div className={home ? 'min-h-dvh bg-[#fbf9f5] text-[#1f1d1a]' : 'flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-300'}>
      {!home && <PanelBoundary name="top bar">
        <TopBar />
      </PanelBoundary>}

      <div className={home ? '' : 'min-h-0 min-w-0 flex-1 overflow-hidden'}>
        <div className={home ? '' : 'h-full min-h-0 overflow-hidden'}>
          <PanelBoundary name="agent studio"><AgentStudio /></PanelBoundary>
        </div>
      </div>

      {voiceOpen && <PanelBoundary name="voice conductor">
        <VoiceConductor />
      </PanelBoundary>}

      {statusOpen && <PanelBoundary name="pulse">
        <Pulse />
      </PanelBoundary>}
      {!home && <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-neutral-800 px-3">
        <button type="button" aria-expanded={voiceOpen} onClick={() => setVoiceOpen(open => !open)} className="inline-flex min-h-11 items-center gap-2 rounded px-3 text-sm text-neutral-400 hover:text-white focus-visible:outline-2 focus-visible:outline-emerald-300"><Mic size={16} aria-hidden="true" />Voice</button>
        <button type="button" aria-expanded={statusOpen} onClick={() => setStatusOpen(open => !open)} className="inline-flex min-h-11 items-center gap-2 rounded px-3 text-sm text-neutral-400 hover:text-white focus-visible:outline-2 focus-visible:outline-emerald-300"><Activity size={16} aria-hidden="true" />System status</button>
      </footer>}
    </div>
  )
}
import { useState } from 'react'
import { Activity, Mic } from 'lucide-react'
