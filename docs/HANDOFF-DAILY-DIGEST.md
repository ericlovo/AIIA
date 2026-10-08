# Daily Digest — Mini enablement

Studio build-out item 2. One scheduled digest: agent/loop lines from #103,
plus a **Repos** section with mounted-repo triage (moved / stuck / CI /
behind-main drift). There is no second `daily_digest` task, no 12:00 UTC
cron, and no dedicated digest agent. Draft PRs **#52**, **#47**, **#105**,
and **#106** were not touched.

## Schedule

Built-in task `daily_digest` fires **daily at 07:40 America/Chicago**.
Daily Brief stays at 07:00 Chicago. Catch-up after a restart is at most
once per local day. Jobs shows `daily 07:40 America/Chicago`.

## What it reads

Agents, launchd loops (`~/.aiia/loops-registry.json`), failing built-in
tasks, and inbox counts by source. The Repos section then reads mounted
checkouts from `REPO_MOUNTS` (`aiia`, `mindmoor`, `sanction`, `proxy-ai`,
plus `morrow` / `mia` if those ids are added later). No `git fetch`.
GitHub reads reuse the existing `gh api` GET adapter. No new egress,
no AIRGAP change.

Mindmoor drift: if `production` or `origin/production` exists, count
commits on `main` that are not on that ref (`git rev-list --count
<branch>..main`). Same for `alumni` / `origin/alumni`, or the first
`release/alumni*` / `origin/release/alumni*` ref when the exact name is
missing.

GitHub remotes with HTTPS userinfo (`x-access-token@github.com`) still
resolve to `owner/repo` only. The token never enters a digest line.

## Output

One inbox row per Chicago date (`digest:<date>`), optionally posted
through the memory-post outbox. Body sections:

- Agents (runs today, waiting review, failed, the day's verdict)
- Loops
- Built-in tasks failing (if any)
- **Repos** — moved PRs/commits, failing CI, merge conflicts, drift
- Inbox waiting review

Quiet/green repo days are just `CLEAR: no material drift/CI` in Repos.
They do not open a Work item and do not record a `quiet_clear` loop
check. Incomplete git/gh reads show as `Incomplete: <repo>` in that
section.

## Mini after merge

1. `git pull` on the Brain checkout (this repo). Do not rebase or merge
   draft PRs #52 / #47 / #105 / #106.
2. Restart Command Center so the built-in task is registered.
3. Optional smoke: `POST /api/tasks/daily_digest/run` and confirm Jobs
   plus the inbox show today's body, including Repos.

No seed script. Leave `agent_data.json` uncommitted. No new
dependencies, no secrets, no AIRGAP / Sanction policy changes.
