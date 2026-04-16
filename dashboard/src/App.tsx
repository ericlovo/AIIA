import { useState } from 'react'
import { Sidebar } from './components/Sidebar'
import { Topbar } from './components/Topbar'
import { BrainOverlay } from './components/BrainOverlay'
import { TodayPage } from './pages/TodayPage'
import { StoriesPage } from './pages/StoriesPage'
import { OpsPage } from './pages/OpsPage'
import { ChatPage } from './pages/ChatPage'

export type Page = 'today' | 'stories' | 'ops' | 'chat'

export default function App() {
  const [page, setPage] = useState<Page>('today')
  const [brainOpen, setBrainOpen] = useState(false)

  // Chat uses the full main area (its own header + thread list), so
  // the outer <main> padding would interfere. Drop padding for chat only.
  const mainClass = page === 'chat' ? 'flex-1 overflow-hidden' : 'flex-1 overflow-y-auto p-6'

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar page={page} onNavigate={setPage} />
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar onBrainToggle={() => setBrainOpen(o => !o)} brainOpen={brainOpen} />
        <main className={mainClass}>
          {page === 'today' && <TodayPage />}
          {page === 'stories' && <StoriesPage />}
          {page === 'ops' && <OpsPage />}
          {page === 'chat' && <ChatPage />}
        </main>
      </div>
      <BrainOverlay open={brainOpen} onClose={() => setBrainOpen(false)} />
    </div>
  )
}
