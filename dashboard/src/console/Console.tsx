import { TopBar } from './TopBar'
import { Pulse } from './Pulse'
import { PanelBoundary } from './ErrorBoundary'
import { AgentStudio } from './AgentStudio'

export function Console() {
  return (
    <div className="h-screen flex flex-col bg-neutral-950 text-neutral-300 overflow-hidden">
      <PanelBoundary name="top bar">
        <TopBar />
      </PanelBoundary>

      <PanelBoundary name="agent studio"><AgentStudio /></PanelBoundary>

      <PanelBoundary name="pulse">
        <Pulse />
      </PanelBoundary>
    </div>
  )
}
