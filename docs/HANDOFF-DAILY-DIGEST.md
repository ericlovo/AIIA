# Daily Digest — Mini enablement

Studio build-out item 2. One scheduled digest: **product status** at the top
(shipped / blocked / waiting on you), then a **Customers** block in the same
form, then a short **Needs a decision** list, then a compressed agent/loop
footer. Repo, CI, and Mindmoor drift collectors from #104 still feed those
lines. There is no second `daily_digest` task, no 12:00 UTC cron, and no
dedicated digest agent. Draft PRs **#52**, **#47**, and **#106** were not
touched.

## Schedule

Built-in task `daily_digest` fires **daily at 07:40 America/Chicago**.
Daily Brief stays at 07:00 Chicago. Catch-up after a restart is at most
once per local day. Jobs shows `daily 07:40 America/Chicago`.

## What it reads

Products, customers, and GitHub slugs come from checked-in
`config/digest_products.json` (override with `AIIA_DIGEST_PRODUCTS`).
Default products: AIIA, Mindmoor (production / alumni drift), Sanction, MIA,
Morrow. Morrow has no repo configured yet. Default customers: That's Right
Sweetie (TRS), Alumni Nations, Smart Medical. Missing mappings stay blank;
Smart Medical prints `not mapped yet`. Never invent a repo, agent, or branch.

For each configured mount the digest reuses the #104 collectors: open / merged
PRs, failing CI on `main`, merge conflicts, check state, and Mindmoor
`production` / `alumni` behind `main` (`git rev-list --count <branch>..main`).
No `git fetch`. GitHub reads reuse the existing `gh api` GET adapter. No new
egress, no AIRGAP change. A missing or unreadable checkout becomes
`no repo mounted` or `repo unreadable` on that product line — it does not
fail the digest.

Studio agents map onto products via `suite`, `memory_namespace`, `repo` /
`repo_id`, or optional `handles` / `kind` when present (#106). Those fields
are optional; a legacy agent without them still maps from suite or repo.

GitHub remotes with HTTPS userinfo (`x-access-token@github.com`) still
resolve to `owner/repo` only. The token never enters a digest line.

## Output

One inbox row per Chicago date (`digest:<date>`), optionally posted
through the memory-post outbox. The inbox row keeps the fuller
`shipped | blocked | waiting on you` layout (empty segments print `none`).
The Slack post is a separate morning-note body: text only, no buttons,
Slack mrkdwn enabled for `digest:` posts.

Slack sections (action-first format):

- Date header, then **Needs attention**: scheduled-task failures, main CI failures,
  conflicts/check failures on active non-dependency PRs, eligible PR numbers to
  review for merge, agent output reviews, and delivery drift. Missing or incomplete
  repository evidence is explicit; it cannot produce an all-clear or merge ask.
- **Decisions**: at most three current inbox suggestions, each on its own line.
  Rows explicitly labeled `Stale story:` become a backlog count instead. No inbox
  records are dismissed, deleted, or marked reviewed by this presentation filter.
- **Merged — deployment not verified**: merged PR numbers, without calling them
  shipped or claiming the deployment is live.
- **Background — not merge requests**: draft/dependency/held PR counts, consulting
  status, and remaining backlog. These PR categories stay outside the merge asks.
- Quiet products share a compact **No active alerts** line. Drafts and maintenance
  remain separate; this is not a claim that every PR passes.
- Failure messages name the task and direct the operator to inspect its Studio
  run before retrying. Raw exception strings stay out of Slack.
- Whole rows fit within the existing message limit, preserving `_Details in Studio_`
  and a “More items in Studio” marker when needed.

Slack counts an awaiting-review agent marker only when no pending output already
represents that agent. The fuller inbox body and Studio morning-note rendering
retain their existing contracts. Slack formatting does not change schedules,
permissions, runtime records, or delivery/outbox deduplication.

Quiet/green product days still print the three inbox segments with
`none`. They do not open a Work item and do not record a `quiet_clear`
loop check.

## Mini after merge

1. `git pull` on the Brain checkout (this repo). Do not rebase or merge
   draft PRs #52 / #47 / #106.
2. Restart Command Center so the built-in task is registered.
3. Optional smoke: `POST /api/tasks/daily_digest/run` and confirm Jobs
   plus the inbox (and Slack, when memory posts are configured) show
   today's product-status body.

No seed script. Leave `agent_data.json` uncommitted. No new
dependencies, no secrets, no AIRGAP / Sanction policy changes.
