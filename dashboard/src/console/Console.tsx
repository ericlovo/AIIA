import { TopBar } from './TopBar'
import { Pulse } from './Pulse'
import { PanelBoundary } from './ErrorBoundary'
import { AgentStudio } from './AgentStudio'
import { VoiceConductor } from './VoiceConductor'

export function Console() {
  return (
    <div className="h-screen flex flex-col bg-neutral-950 text-neutral-300 overflow-hidden">
      <PanelBoundary name="top bar">
        <TopBar />
      </PanelBoundary>

      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
        <div className="h-full min-h-0 overflow-hidden">
          <PanelBoundary name="agent studio"><AgentStudio /></PanelBoundary>
        </div>
      </div>

      <PanelBoundary name="voice conductor">
        <VoiceConductor />
      </PanelBoundary>

      <PanelBoundary name="pulse">
        <Pulse />
      </PanelBoundary>
    </div>
  )
}
