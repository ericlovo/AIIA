/**
 * Chat store — localStorage persistence for chat identity and threads.
 *
 * Schema is versioned (`version: 1`) so future migrations can detect and
 * either upgrade or discard old payloads rather than crashing on unexpected
 * shapes. Per Vercel's `client-localstorage-schema` rule.
 */

export type TeamUser = 'Eric' | 'Paul' | 'Tony'

export const TEAM_USERS: TeamUser[] = ['Eric', 'Paul', 'Tony']

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  createdAt: string
}

export interface ChatThread {
  id: string
  title: string
  createdAt: string
  messages: ChatMessage[]
}

interface StoredV1 {
  version: 1
  user: TeamUser | null
  threads: ChatThread[]
  activeThreadId: string | null
}

const STORAGE_KEY = 'aiia.chat.v1'

const EMPTY: StoredV1 = {
  version: 1,
  user: null,
  threads: [],
  activeThreadId: null,
}

export function loadChatState(): StoredV1 {
  if (typeof window === 'undefined') return EMPTY
  const raw = window.localStorage.getItem(STORAGE_KEY)
  if (!raw) return EMPTY
  try {
    const parsed = JSON.parse(raw) as Partial<StoredV1>
    if (parsed.version !== 1) return EMPTY
    return {
      version: 1,
      user: parsed.user ?? null,
      threads: Array.isArray(parsed.threads) ? parsed.threads : [],
      activeThreadId: parsed.activeThreadId ?? null,
    }
  } catch {
    return EMPTY
  }
}

export function saveChatState(state: StoredV1): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // Quota exceeded or storage disabled — drop silently. Chat still works
    // in-memory for the session, just won't persist across reloads.
  }
}

export function newThreadId(): string {
  return `t_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

export function deriveThreadTitle(firstMessage: string): string {
  const trimmed = firstMessage.trim().replace(/\s+/g, ' ')
  if (trimmed.length <= 40) return trimmed
  return trimmed.slice(0, 40).trimEnd() + '…'
}
