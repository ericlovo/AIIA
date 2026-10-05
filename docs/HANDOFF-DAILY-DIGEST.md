# Daily Digest — Mini enablement

Studio build-out item 2. Deterministic daily triage of mounted repos (moved /
stuck / CI / behind-main drift). Separate from Daily Brief (`daily_brief`,
08:00 UTC, LLM memory write). Draft PRs **#52** and **#47** were not touched.

## Schedule

Built-in task `daily_digest` fires **daily at 12:00 UTC**.

That is **07:00 America/Chicago during CDT** and **06:00 during CST**. The
task scheduler is UTC-only (hour + minute), so the cron does not shift with
DST. Jobs shows `daily 12:00 UTC`.

## What it reads

Mounted checkouts from `REPO_MOUNTS` (`aiia`, `mindmoor`, `sanction`,
`proxy-ai`, plus `morrow` / `mia` if those ids are added later). No `git
fetch`. GitHub reads reuse the existing `gh api` GET adapter.

Mindmoor drift: if `production` or `origin/production` exists, count commits
on `main` that are not on that ref. Same for `alumni` / `origin/alumni`, or
the first `release/alumni*` / `origin/release/alumni*` ref when the exact
name is missing.

## Output

Exactly one line (≤ 200 chars), for example:

- `CLEAR: no material drift/CI`
- `Moved: AIIA 2 PRs | Stuck: AIIA CI | Drift: mindmoor production −12 behind main`

Delivered through the #101 output-channel path. Default channel is
`studio_inbox`. Slack is a declared destination only: if the agent is set to
`slack` and outbound is missing or posting is still pending, the line goes to
the Studio inbox with the existing note.

## Quiet vs review

- **CLEAR / green:** record a `quiet_clear` loop check and the agent's last
  result. No Work item. `pending_loop_reviews` stays 0, so the next day is
  not blocked.
- **Stuck** (failing CI, merge conflicts, behind-main / production-alumni
  drift): one interval Work item. Same fingerprint while that item is still
  unreviewed is history only (`verified_unchanged`), not a second inbox row.
  At the existing review cap the task still runs and records
  `awaiting_review` without opening another item.
- **Incomplete** (a mounted repo's git/gh read failed): `loop_check` failure.
  Those do not count toward the review cap, so the next day still runs.

## Mini after merge

1. `git pull` on the Brain checkout (this repo). Do not rebase or merge draft
   PRs #52 / #47.
2. Restart Command Center (`com.aiia.command-center` / the :8200 process) so
   the new built-in task is registered. Daily Brief is unchanged.
3. Enable the standing agent (does **not** write a committed runtime file):

   ```bash
   python -m local_brain.scripts.ensure_daily_digest_agent
   python -m local_brain.scripts.ensure_daily_digest_agent --apply
   # optional: --channel slack   # still falls back to inbox until posting exists
   ```

   Loop stays **off**. The built-in task owns the cadence. Leave `agent_data.json`
   uncommitted.

4. Optional smoke: `POST /api/tasks/daily_digest/run` and confirm Jobs shows
   today's line. A CLEAR result should not appear as Work awaiting review.

No new dependencies, no secrets, no AIRGAP / Sanction policy changes.
GitHub read uses the Mini's existing allowlisted `gh` path.
