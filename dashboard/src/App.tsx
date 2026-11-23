import { Console } from './console/Console'
import { Desk } from './console/Desk'
import { BDRecord } from './console/BDRecord'
import { formatRoute } from './console/studioRoute'
import { useStudioRoute } from './console/useStudioRoute'

function LocalDesk() {
  const { route } = useStudioRoute()
  return <div className="flex h-dvh flex-col">
    <nav aria-label="Studio" className="flex shrink-0 gap-4 px-4">
      {(['desk', 'record'] as const).map(view => <a key={view} href={formatRoute({ view })}
        aria-current={view === route.view ? 'page' : undefined}
        className="inline-flex min-h-11 items-center text-sm aria-[current=page]:underline">
        {view === 'desk' ? 'Desk' : 'Record'}
      </a>)}
    </nav>
    <div className="min-h-0 flex-1">
      {route.view === 'record' ? <BDRecord receiptId={route.receiptId} showRun={route.showRun} />
        : route.view === 'desk' ? <Desk /> : <p>Open Desk or Record to review synthetic research.</p>}
    </div>
  </div>
}

export default function App() {
  return document.querySelector('meta[name="studio-desk-dev"]') ? <LocalDesk /> : <Console />
}
