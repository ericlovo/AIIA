import type { AgentDefinition } from '../lib/api'

export interface DevelopmentLoop {
  id: string
  name: string
  output: string
  cadence: string
  draft: AgentDefinition
}

function loop(name: string, mission: string, task: string, minutes: number, cap: number, repo = 'aiia'): AgentDefinition {
  return {
    name, mission, persona: 'Evidence first. Separate observed facts, unknowns, and recommendations. Never claim tests or deployments ran without supplied evidence.',
    skills: ['Analysis', 'Coding'], tools: ['Repository read'], repo_id: repo,
    temperature: 0.2, max_tokens: 1600, loop_enabled: false,
    loop_interval_minutes: minutes, loop_max_runs_per_day: cap,
    loop_task: `${task}\nUse only the supplied repository snapshot. Cite commit IDs or visible paths. Mark missing code, CI results, or prior snapshots as unknown. Do not invent changes since the previous run. Return a concise markdown artifact with Evidence, Findings, and Next action. All writes, test execution, and publication require a human approval.`,
  }
}

export const DEVELOPMENT_LOOPS: DevelopmentLoop[] = [
  { id: 'change-radar', name: 'Change Radar', output: 'Change risk report', cadence: 'Every 4h / 3 runs per day', draft: loop('Change Radar', 'Triage the risk in recent AIIA repository changes.', 'Inspect recent commits and working-tree summaries. Rank up to three changes by regression risk, name what evidence is missing, and propose the next focused review.', 240, 3) },
  { id: 'regression-planner', name: 'Regression Planner', output: 'Test plan', cadence: 'Every 8h / 2 runs per day', draft: loop('Regression Planner', 'Turn recent changes into a bounded regression plan.', 'Using the visible changed paths and repository layout, propose up to five regression scenarios with setup, trigger, and expected outcome. Distinguish test proposals from tests known to exist. Do not claim coverage percentages or passing tests.', 480, 2) },
  { id: 'release-review', name: 'Release Review', output: 'Release readiness brief', cadence: 'Every 12h / 1 run per day', draft: loop('Release Review', 'Prepare an evidence-backed release handoff.', 'Summarize visible recent commits into release notes. List unresolved working-tree changes, unverified CI gates, and rollback questions. Return READY FOR HUMAN REVIEW or BLOCKED with reasons. Never claim a release was deployed.', 720, 1) },
  { id: 'mindmoor-local', name: 'Mindmoor Local Planner', output: 'Local job proposal', cadence: 'Every 12h / 1 run per day', draft: loop('Mindmoor Local Planner', 'Identify bounded development jobs suitable for the Mini.', 'Review the supplied Mindmoor repository snapshot. Propose one local development job with inputs, expected artifact, timeout, retry limit, and sensitivity boundary. Cite evidence and explicitly identify missing cron definitions or source content.', 720, 1, 'mindmoor') },
]
