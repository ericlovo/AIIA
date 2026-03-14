import { useState } from 'react'
import { Sidebar } from './components/Sidebar'
import { Topbar } from './components/Topbar'
import { BrainOverlay } from './components/BrainOverlay'
import { TodayPage } from './pages/TodayPage'
import { StoriesPage } from './pages/StoriesPage'
import { OpsPage } from './pages/OpsPage'

export type Page = 'today' | 'stories' | 'ops'

export default function App() {
  const [page, setPage] = useState<Page>('today')
  const [brainOpen, setBrainOpen] = useState(false)

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar page={page} onNavigate={setPage} />
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar onBrainToggle={() => setBrainOpen(o => !o)} brainOpen={brainOpen} />
        <main className="flex-1 overflow-y-auto p-6">
          {page === 'today' && <TodayPage />}
          {page === 'stories' && <StoriesPage />}
          {page === 'ops' && <OpsPage />}
        </main>
      </div>
      <BrainOverlay open={brainOpen} onClose={() => setBrainOpen(false)} />
    </div>
  )
}
