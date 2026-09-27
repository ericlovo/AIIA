# Review bots: what runs, what it costs, what it catches

Evaluated 2026-09-27 across ericlovo/aiia#40 to #77.

## Findings

**Verified (GitHub history):**

| Bot | Runs | Real findings | State |
|---|---|---|---|
| Cursor Bugbot | about 24 (20 on head commits since Sep 17) | 2, both on #56 | Every run since #58 (Sep 17) ends "usage limit reached" |
| Cursor Security Agent (Automation) | 24, 3.5 to 6.5 min each | 1: unauthenticated `/api/memory-inbox/ingest` (#66) | Runs on every PR, including Dependabot |
| Cursor Approval Agent (Automation) | 24 | 0 | Approved #68, which broke the MCP server import and was closed. Inconsistent verdicts on identical inputs |
| CodeRabbit | 0 | 0 | Never posted on this repo. `.coderabbit.yaml` had `enable_free_tier: false` |
| Own CI (ruff, pytest, bandit, pip-audit, gitleaks, browser suites) | every push | caught the rotted browser suites (#74) | Green, free |

**Inferred, not verified** (Cursor's billing pages are not reachable from
this environment):

- Since May 2026 Bugbot bills per run from the plan's included usage, at
  about $1.00 to $1.50 per run, instead of a flat seat fee.
- Cursor Automations bill as cloud-agent usage. Private or Team Visible
  automations bill to the person who created them.
- Therefore about three Cursor cloud-agent runs per PR (Bugbot, Security,
  Approval) drew from one Pro allowance. The limit hit five days after all
  three were enabled.

**Unknown:** whether the CodeRabbit GitHub App is installed on this repo.

## Decisions

| Bot | Role | Trigger |
|---|---|---|
| CodeRabbit | Primary line-level PR reviewer plus ruff, eslint, semgrep, gitleaks, shellcheck | Every PR (free on a public repo) |
| Cursor Bugbot | Second opinion on high-risk PRs only | On request: comment `bugbot run` |
| Cursor Security Agent | Security pass on the security boundary | Only when the PR touches `local_brain/execution/`, `local_brain/egress.py`, `local_brain/command_center/`, or `.github/workflows/`; skip Dependabot |
| Cursor Approval Agent | Retire | Off |
| Own CI | The merge gate | Every push |

Why retire the Approval Agent: it produced no findings, approved a
breaking change, and humans (or Claude Approvals, if adopted) are the
merge gate. Why keep the Security Agent narrowly: it produced the single
most important finding in the window.

## Settings only the account owner can change

These live in the vendors' dashboards, not the repo.

**Cursor** (Dashboard → Bugbot, and Dashboard → Automations):

1. Bugbot: enable **Run only when mentioned**. Keep draft PRs off.
2. Automations → Approval Agent: disable.
3. Automations → Security Reviewer: set the trigger to PR opened (not every
   push) and add path filters as in the table above. Exclude Dependabot.
4. Check each automation's ownership. A Private automation bills your own
   Pro usage; a Team Owned one bills the team pool.
5. Usage page: confirm what consumed the pool since Sep 12.

**CodeRabbit:**

1. Install the CodeRabbit GitHub App on `ericlovo/AIIA` (GitHub → Settings →
   Applications, or from the CodeRabbit dashboard) if it is not listed.
2. After this repo change merges, open any PR and confirm a CodeRabbit
   check appears. If none does, comment `@coderabbitai review`.

## What moves to the M4 Agent Studio

The Mini already has the pieces: `code_review` is an accepted proposal
source in the memory inbox, and review outcomes (`needs_work`,
`already_fixed`, `declined`, `external_failure`) are tracked on Today.

1. **Now: measure the vendors.** A loop files each bot finding into the
   inbox as a `code_review` proposal with the bot's name. Triaging it
   records the outcome, so review health shows each vendor's real
   precision. Quota and plan failures file as `external_failure`, never as
   findings.
2. **Next: a local first pass** for anything that must not leave the box:
   deterministic tools (ruff, bandit, semgrep) plus a local model checking
   the invariants in `.cursor/BUGBOT.md`, on each new PR head, filing into
   the same inbox. It runs under air-gap, costs no tokens, and is the only
   option for tenant code that cannot go to a vendor.
3. **Keep vendors for** broad line-level review on the public repo, where
   they are free or cheap and the code is already public.

## Open security finding

From #66, verified 2026-09-27: Command Center (`command_center/server.py`)
has no authentication, only CORS, and binds `0.0.0.0:8200`. CORS does not
stop non-browser clients. Anyone who can reach the Mini on the network can
call the ingest endpoint, and also the approve endpoints for actions, git
workspaces and git writes, and agent runs. Tracked separately; see the
session notes for the proposed fix.
