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

Slack sections:

- Header `*AIIA · Thu Oct 8*`
- One product line each, one status emoji (red = blocked, yellow =
  waiting on Eric, green = shipped or clean, white = not mapped / no
  repo). Empty segments are omitted. Bold only the name and
  `waiting on you`.
- `*Customers*` in the same form. TRS shows Mindmoor `main` commits not
  yet on its tenant/branch when that ref exists. Alumni Nations shows
  alumni drift, review waiting on Alumni Nations Research Scout, and
  days to Phase 1 kickoff (2026-10-15) or days into the phase (through
  2027-01-12). Smart Medical is `not mapped yet`.
- `*Needs you (N)*: title · title` when there is something to decide.
  Omitted when empty — never `none`.
- One short failure line only when a built-in or agent failed. No
  agent, loop, run, or inbox counts.
- `_Details in Studio_`

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
