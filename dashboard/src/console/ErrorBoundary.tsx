import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  name: string
  children: ReactNode
}

interface State {
  error: Error | null
}

// Per-panel error boundary so a bug in one lane can't take down the whole
// console. The message is intentionally terse — enough to know which panel
// crashed and what the error was, not a stack trace users don't want.
export class PanelBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error(`[PanelBoundary] ${this.props.name} crashed:`, error, info)
  }

  handleRetry = () => {
    this.setState({ error: null })
  }

  render() {
    if (this.state.error) {
      return (
        <div className="h-full flex items-center justify-center p-6 bg-[#0a0a0a]">
          <div className="text-center max-w-xs">
            <div className="text-[10px] text-red-400 tracking-[0.3em] font-semibold mb-2">
              {this.props.name.toUpperCase()} CRASHED
            </div>
            <p className="text-xs text-[#888] mb-4 leading-relaxed">
              {this.state.error.message || 'Something went wrong rendering this panel.'}
            </p>
            <button
              onClick={this.handleRetry}
              className="text-xs px-3 py-1.5 rounded-lg border border-[#2a2a2a] text-[#888] hover:text-white hover:border-purple-500/50 cursor-pointer"
            >
              Retry
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
