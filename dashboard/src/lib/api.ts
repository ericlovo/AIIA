const BASE = '';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// Types
export interface Service {
  id: string;
  name: string;
  category: string;
  status: 'online' | 'degraded' | 'offline';
  response_time_ms: number;
  avg_response_time_ms: number;
  uptime_pct: number;
  total_checks: number;
  error_count: number;
  consecutive_up: number;
  sparkline: { ms: number; ok: boolean }[];
}

export interface Action {
  id: string;
  type: string;
  severity: string;
  title: string;
  description: string;
  proposed_fix: string;
  source_task: string;
  status: string;
  files_affected: string[];
  created_at: string;
  rejected_reason: string | null;
}

export interface Story {
  id: string;
  title: string;
  product: string;
  priority: string;
  status: string;
  description: string;
  source_type?: string;
  tags?: string[];
  client_impact?: string;
  priority_score?: number;
  composite_score?: number;
  geometric?: { alignment: number; magnitude: number; geometric_score: number };
  priority_reasoning?: string;
  filter_scores?: Record<string, number>;
  suggested_priority?: string;
  created_at: string;
  updated_at: string;
}

export interface Commit {
  hash: string;
  subject: string;
  author: string;
  type: string;
  category: string;
  files: string[];
  product?: string;
}

export interface WipEntry {
  id: string;
  fact: string;
  source: string;
  created_at: string;
}

export interface CheckinData {
  timestamp: string;
  wip: WipEntry[];
  recent_sessions: { id: string; fact: string; created_at: string }[];
  recent_commits: { total: number; by_product: Record<string, number>; commits: Commit[] };
  active_stories: Story[];
  blocked_stories: Story[];
  pending_actions: { total: number; by_severity: Record<string, number> };
  pipeline: unknown[];
  roadmap_summary: { total: number; by_priority: Record<string, number>; by_status: Record<string, number> };
}

export interface WorkContext {
  today: {
    date: string;
    summary: {
      total_commits: number;
      total_files_changed: number;
      total_additions: number;
      total_deletions: number;
      products_touched: number;
      authors: string[];
    };
    products: Record<string, { commit_count: number; commits: Commit[] }>;
  };
}

export interface TaskInfo {
  id: string;
  name: string;
  description: string;
  interval_seconds: number;
  last_run: string | null;
  next_run: string | null;
  last_status: string | null;
  run_count: number;
  enabled: boolean;
}

export interface ExecutionStatus {
  enabled: boolean;
  is_running: boolean;
  kill_switch: boolean;
  active_subprocesses: number;
  recent: {
    id: string;
    action_type: string;
    strategy: string;
    safety_tier: string;
    status: string;
    duration_ms: number;
    input_summary: string;
    output_summary: string;
    started_at: string;
  }[];
  stats: {
    total: number;
    by_status: Record<string, number>;
    success_rate: number;
  };
}

// API calls
export const api = {
  health: () => get<{ aiia: { status: string }; ollama: { status: string } }>('/api/health'),
  checkin: () => get<CheckinData>('/api/checkin'),
  workContext: () => get<WorkContext>('/api/work/context'),
  monitor: () => get<{ services: Record<string, Service> }>('/api/monitor'),

  actions: (status?: string) => {
    const q = status ? `?status=${status}` : '';
    return get<{ actions: Action[]; summary: Record<string, unknown> }>(`/api/actions${q}`);
  },
  actionsSummary: () => get<{ total: number; by_status: Record<string, number>; pending_by_severity?: Record<string, number> }>('/api/actions/summary'),
  approveAction: (id: string) => post<{ approved: boolean }>(`/api/actions/${id}/approve`),
  rejectAction: (id: string, reason: string) => post<{ rejected: boolean }>(`/api/actions/${id}/reject`, { reason }),

  stories: (status?: string) => {
    const q = status ? `?status=${status}` : '';
    return get<{ stories: Story[]; count: number }>(`/api/roadmap${q}`);
  },
  storySummary: () => get<{ total: number; by_priority: Record<string, number>; by_status: Record<string, number>; by_product: Record<string, number> }>('/api/roadmap/summary'),
  updateStory: (id: string, data: Partial<Story>) =>
    fetch(`/api/roadmap/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(r => r.json()),
  prioritize: (limit = 10) => post<{ stories: Story[]; count: number }>('/api/roadmap/prioritize', { limit }),

  tasks: () => get<TaskInfo[]>('/api/tasks'),
  runTask: (id: string) => post<{ started: boolean }>(`/api/tasks/${id}/run`),
  executionStatus: () => get<ExecutionStatus>('/api/execution/status'),

  briefingLatest: () => get<{ briefing: string; generated_at: string; source: string }>('/api/briefing/latest'),
  tokensToday: () => get<Record<string, unknown>>('/api/tokens/today'),

  // Brain overlay
  memories: () => get<{ memories: { id: string; fact: string; category: string; created_at: string }[] }>('/api/memories'),
  search: (question: string) => post<{ results: { content: string; source: string; score?: number }[] }>('/api/chat', { message: question, mode: 'text' }),
};
