const BASE = '';

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const payload = await res.json().catch(() => null) as { detail?: string } | null;
    const detail = payload?.detail;
    throw new Error(detail === 'mini_busy' ? 'Mini busy — wait for the active run to finish.' : detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  return parse<T>(res);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  return parse<T>(res);
}

async function put<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  return parse<T>(res);
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' });
  return parse<T>(res);
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
  status?: string;
  run_history?: { status: string; summary?: string }[];
  task_id: string;
  name: string;
  description: string;
  interval_seconds: number;
  last_run: string | null;
  next_run: string | null;
  last_status: string | null;
  last_result?: string | null;
  run_count: number;
  fail_count: number;
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

export interface Agent {
  id: string;
  name: string;
  mission: string;
  persona: string;
  skills: string[];
  tools: string[];
  repo_id: string;
  temperature: number;
  max_tokens: number;
  loop_enabled: boolean;
  loop_interval_minutes: number;
  loop_task: string;
  loop_max_runs_per_day: number;
  loop_runs_today: number;
  loop_day: string;
  suite?: string;
  memory_namespace?: string;
  status: 'idle' | 'running' | 'error';
  last_run_at: string | null;
  last_result: string;
  last_error: string;
  runs: AgentRun[];
  created_at: string;
  updated_at: string;
}

export interface AgentRun {
  id?: string;
  task: string;
  result: string;
  error: string;
  at: string;
  trigger?: 'manual' | 'interval' | 'assignment';
  assignment_id?: string;
  model?: string;
  latency_ms?: number;
}

export interface StudioRun {
  id: string;
  agent_id: string;
  agent_name: string;
  repo_id: string;
  at: string;
  status: 'completed' | 'failed';
  trigger: string;
  assignment_id: string;
  model: string;
  latency_ms: number;
  legacy: number;
  task?: string;
  result?: string;
  error?: string;
}

export interface StudioActivity {
  agent_days: { agent_id: string; day: string; total: number; failed: number }[];
  days: { day: string; total: number; completed: number; failed: number; latency_ms: number }[];
  runs: StudioRun[];
  matching: number;
  total: number;
  earliest: string | null;
  imported: number;
  start: string;
  today: string;
  timezone: string;
}

export interface RepositoryResource {
  id: string;
  name: string;
  path: string;
  branch: string;
  dirty: boolean;
  github_repo: string;
  git_workspace?: {
    eligible: boolean;
    reason: string;
  };
}

export interface GitHubResource {
  status: 'connected' | 'disconnected';
  mode: 'read_only';
  provider: string;
  account: string;
  reason: string;
}

export interface AgentSuiteMember {
  slug: string;
  name: string;
  depth: number;
  role: string;
  recommended_max_tokens: number;
  produced_artifacts: string[];
  agents: Array<{
    id: string;
    name: string;
    suite: string;
    memory_namespace: string;
    max_tokens?: number;
    matched_by: 'tag' | 'alias';
  }>;
}

export interface AgentSuite {
  slug: string;
  name: string;
  memory_namespace: string;
  memory_source: string;
  repository_id: string;
  max_depth: number;
  members: AgentSuiteMember[];
}

export type AgentDefinition = Pick<Agent,
  'name' | 'mission' | 'persona' | 'skills' | 'tools' | 'repo_id' |
  'temperature' | 'max_tokens' | 'loop_enabled' | 'loop_interval_minutes' |
  'loop_task' | 'loop_max_runs_per_day'
> & {
  suite?: string;
  memory_namespace?: string;
};

export type AssignmentStatus = 'queued' | 'running' | 'completed' | 'failed';
export type ReviewStatus = 'unreviewed' | 'accepted' | 'rejected';
export type AssignmentPriority = 'low' | 'normal' | 'high' | 'urgent';

export interface Assignment {
  recovery_pending?: boolean;
  recovered_at?: string;
  id: string;
  title: string;
  objective: string;
  agent_id: string;
  priority: AssignmentPriority;
  context: string;
  success_criteria: string;
  source_handoff_id: string;
  review_status?: ReviewStatus;
  review_note?: string;
  reviewed_at?: string | null;
  review_version?: string;
  status: AssignmentStatus;
  result: string;
  error: string;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export type AssignmentDefinition = Pick<Assignment,
  'title' | 'objective' | 'agent_id' | 'priority' | 'context' | 'success_criteria'
>;

export type HandoffArtifactType = 'brief' | 'analysis' | 'plan' | 'decision' | 'review';

export interface Handoff {
  id: string;
  source_assignment_id: string;
  target_assignment_id: string;
  from_agent_id: string;
  to_agent_id: string;
  artifact_type: HandoffArtifactType;
  artifact: string;
  instructions: string;
  status: AssignmentStatus;
  created_at: string;
  updated_at: string;
}

export interface HandoffDefinition {
  source_assignment_id: string;
  to_agent_id: string;
  artifact_type: HandoffArtifactType;
  instructions: string;
}

export interface AgentWorldPoint {
  x: number;
  y: number;
}

export interface AgentWorldLayout {
  version: 1;
  revision: number;
  positions: Record<string, AgentWorldPoint>;
  updated_at: string | null;
}

export type GitWriteOp = 'write_file' | 'run_tests' | 'commit' | 'push' | 'open_pr';
export type GitWriteStatus = 'pending' | 'approved' | 'completed' | 'failed' | 'rejected';

export interface GitWrite {
  id: string;
  workspace_id: string;
  assignment_id: string;
  op: GitWriteOp;
  title: string;
  status: GitWriteStatus;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  error: string;
  created_at: string;
  updated_at: string;
  approved_at: string | null;
  events: { at: string; action: string; detail: string }[];
}

export interface GitWriteDefinition {
  op: GitWriteOp;
  payload: Record<string, unknown>;
  title?: string;
}

export type GitWorkspaceStatus = 'pending' | 'preparing' | 'ready' | 'failed';

export type VoiceStatus = 'connected' | 'not_configured'

export interface VoiceToolDefinition {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface VoiceSpecialty {
  slug: string
  version: string
  name: string
  depth: number
  recommendedDepth: number
  maxDelegationDepth: number
  authority: string
  requiredTools: string[]
  producedArtifacts: string[]
  requiredApprovalLevel: string
}

export interface VoiceStatusResponse {
  status: VoiceStatus
  configured: boolean
  provider: string
  model: string
  voice: string
  realtime_url: string
  reason: string
  tools: VoiceToolDefinition[]
  allowlist: string[]
  forbidden: string[]
  specialty: VoiceSpecialty
}

export interface VoiceSessionResponse {
  status: VoiceStatus
  token: string
  expires_at: number | string | null
  ttl_seconds: number
  realtime_url: string
  session: {
    voice: string
    instructions: string
    turn_detection: null
    tools: VoiceToolDefinition[]
    audio: Record<string, unknown>
  }
  specialty: VoiceSpecialty
}

export interface VoiceToolResult {
  ok: boolean
  name: string
  result?: Record<string, unknown>
  error?: string
}

export interface GitWorkspace {
  id: string;
  assignment_id: string;
  agent_id: string;
  repo_id: string;
  title: string;
  status: GitWorkspaceStatus;
  branch: string;
  base_ref: string;
  path: string;
  git_status: string;
  error: string;
  created_at: string;
  updated_at: string;
  approved_at: string | null;
  events: { at: string; action: string; detail: string }[];
}

// API calls
export const api = {
  studioActivity: (agentId = '', day = '', status = '') =>
    get<StudioActivity>(`/api/studio/activity?${new URLSearchParams({ agent_id: agentId, day, status })}`),
  studioRun: (id: string) => get<{ run: StudioRun }>(`/api/studio/runs/${encodeURIComponent(id)}`),
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
  createStory: (data: { title: string; product?: string; priority?: string; status?: string; description?: string; tags?: string[]; client_impact?: string; source_type?: string }) =>
    post<{ story: Story }>('/api/roadmap', data),
  updateStory: (id: string, data: Partial<Story>) =>
    put<{ story: Story }>(`/api/roadmap/${id}`, data),
  deleteStory: (id: string) =>
    del<{ deleted: boolean }>(`/api/roadmap/${id}`),
  prioritize: (limit = 10) => post<{ stories: Story[]; count: number }>('/api/roadmap/prioritize', { limit }),

  tasks: () => get<TaskInfo[]>('/api/tasks'),
  runTask: (id: string) => post<{ started: boolean }>(`/api/tasks/${id}/run`),
  executionStatus: () => get<ExecutionStatus>('/api/execution/status'),

  agents: () => get<{ agents: Agent[] }>('/api/agents'),
  agentSuites: () => get<{ suites: AgentSuite[] }>('/api/agent-suites'),
  agentResources: () => get<{ repos: RepositoryResource[]; github: GitHubResource }>('/api/agents/resources'),
  createAgent: (data: AgentDefinition) =>
    post<{ agent: Agent }>('/api/agents', data),
  setAgentLoop: (id: string, enabled: boolean) =>
    post<{ agent: Agent }>(`/api/agents/${id}/loop`, { enabled }),
  updateAgent: (id: string, data: AgentDefinition) =>
    put<{ agent: Agent }>(`/api/agents/${id}`, data),
  deleteAgent: (id: string) => del<{ deleted: boolean }>(`/api/agents/${id}`),
  runAgent: (id: string, task: string) =>
    post<{ agent: Agent; model: string; latency_ms: number }>(`/api/agents/${id}/run`, { task }),

  assignments: () => get<{ assignments: Assignment[] }>('/api/assignments'),
  reviewAssignment: (id: string, decision: ReviewStatus, expected_version: string, note: string) =>
    post<{ assignment: Assignment }>(`/api/assignments/${id}/review`, { decision, expected_version, note }),
  createAssignment: (data: AssignmentDefinition) =>
    post<{ assignment: Assignment }>('/api/assignments', data),
  deleteAssignment: (id: string) =>
    del<{ deleted: boolean }>(`/api/assignments/${id}`),
  runAssignment: (id: string) =>
    post<{ assignment: Assignment; agent: Agent; model: string; latency_ms: number }>(`/api/assignments/${id}/run`),
  recoverAssignment: (id: string) =>
    post<{ assignment: Assignment }>(`/api/assignments/${id}/recover`),
  assignmentHistory: (id: string, offset = 0) =>
    get<{ runs: StudioRun[]; total: number; offset: number; limit: number; attempt_id: string; current_output_saved: boolean; completed_run_id: string }>(`/api/assignments/${encodeURIComponent(id)}/history?offset=${offset}`),
  gitWorkspaces: () => get<{ workspaces: GitWorkspace[] }>('/api/git-workspaces'),
  requestGitWorkspace: (assignmentId: string) =>
    post<{ workspace: GitWorkspace }>(`/api/assignments/${assignmentId}/git-workspace`),
  approveGitWorkspace: (workspaceId: string) =>
    post<{ workspace: GitWorkspace }>(`/api/git-workspaces/${workspaceId}/approve`),
  gitWrites: (workspaceId?: string) => {
    const q = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : '';
    return get<{ writes: GitWrite[] }>(`/api/git-writes${q}`);
  },
  proposeGitWrite: (workspaceId: string, data: GitWriteDefinition) =>
    post<{ write: GitWrite }>(`/api/git-workspaces/${workspaceId}/writes`, data),
  approveGitWrite: (writeId: string) =>
    post<{ write: GitWrite }>(`/api/git-writes/${writeId}/approve`),
  rejectGitWrite: (writeId: string, reason = '') =>
    post<{ write: GitWrite }>(`/api/git-writes/${writeId}/reject`, { reason }),
  handoffs: () => get<{ handoffs: Handoff[] }>('/api/handoffs'),
  createHandoff: (data: HandoffDefinition) =>
    post<{ handoff: Handoff; assignment: Assignment }>('/api/handoffs', data),
  deleteHandoff: (id: string) => del<{ deleted: boolean }>(`/api/handoffs/${id}`),
  agentWorldLayout: () => get<{ layout: AgentWorldLayout }>('/api/agent-world/layout'),
  updateAgentWorldLayout: (positions: Record<string, AgentWorldPoint>) =>
    put<{ layout: AgentWorldLayout }>('/api/agent-world/layout', { positions }),
  resetAgentWorldLayout: () => del<{ layout: AgentWorldLayout }>('/api/agent-world/layout'),

  voiceStatus: () => get<VoiceStatusResponse>('/api/voice/status'),
  voiceSession: () => post<VoiceSessionResponse>('/api/voice/session'),
  voiceTool: (name: string, arguments_: Record<string, unknown> = {}) =>
    post<VoiceToolResult>('/api/voice/tools', { name, arguments: arguments_ }),

  briefingLatest: () => get<{ briefing: string; generated_at: string; source: string }>('/api/briefing/latest'),
  tokensToday: () => get<Record<string, unknown>>('/api/tokens/today'),

  // Brain overlay
  memories: () => get<{ memories: { id: string; fact: string; category: string; created_at: string }[] }>('/api/memories'),
  search: (question: string) => post<{ results: { content: string; source: string; score?: number }[] }>('/api/chat', { message: question, mode: 'text' }),
};
