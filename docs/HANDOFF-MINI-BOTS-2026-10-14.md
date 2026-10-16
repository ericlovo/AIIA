# Mini handoff: deploy Studio Bots

## Objective and boundaries

Deploy PR #112 on the Mini, verify the real Studio UI and local services, and
report the running revision. The owner requested this deployment. Read the Mini's
root AGENTS.md and preserve local work before making changes.

Keep airgap enabled and existing egress policy intact. Do not change models,
download models, enable Jev/cloud calls, merge other PRs, alter schedules, or edit
runtime agent/memory data. Those are separate decisions. No destructive Git
operations, force pushes, or changes to `.planning/` or `.claude/`.

## Monitor snapshot preservation before the public-hygiene update

The later public-hygiene change removes
`local_brain/command_center/monitor_data.json` from Git and ignores that path.
The runtime storage path and persistence behavior do not change. A normal pull
can remove the formerly tracked copy: `.gitignore` alone does not preserve it.
Complete this procedure on each existing installation before updating across
that removal; do not merge into an automatically deployed branch until ready.

1. Pause the actual auto-deploy job and stop the Command Center through its
   confirmed service manager, allowing its shutdown persistence to finish.
2. Inspect local changes. Preserve unrelated work separately; do not reset,
   clean, or stash the whole checkout. If the snapshot exists, record its SHA-256
   and move it into a new, private backup directory outside all Git checkouts.
   Verify the moved file matches that hash. Keep the exact backup path locally;
   do not upload the snapshot or its contents to a PR.
3. Apply the normal fast-forward update. If Git refuses, stop and resolve the
   reported state without discarding local data. The external snapshot remains
   the recovery source even if the update fails.
4. Before restarting, copy the preserved snapshot back to the same runtime path,
   preserving its permissions. Verify its hash equals the pre-update hash and
   that it is ignored and absent from `git ls-files`. Keep the external backup.
   If no snapshot existed, do not install the historical repository copy as seed
   data; the monitor supports starting without a persisted snapshot.
5. Restart the confirmed service, verify it loads the preserved monitoring state,
   then resume auto-deploy. If the update failed, restore the preserved file before
   restarting the prior revision instead. No agent, assignment, memory, or SQLite
   data is changed by this procedure.

This is an operator prerequisite, not a migration already performed on the Mini.

## Verified ground state

Checked on 2026-10-14 from the development checkout:

- PR: https://github.com/ericlovo/AIIA/pull/112
- Branch: `codex/studio-bots`; application commit
  `28b232988cb37ce3f9cf8b77adfd6956cc9d0b59`.
- PR is OPEN, draft, MERGEABLE, merge state CLEAN; no recorded approval verdict.
- All six GitHub checks passed on that application commit. CI run
  https://github.com/ericlovo/AIIA/actions/runs/38610692902 reports **1,109 backend
  tests passed, 9 skipped; 85 UI unit tests passed; all 19 browser suites passed**.
  Separate gitleaks, Bandit and dependency audit checks passed too.
- Fetched `origin/main` is `e5a3ab9d84e5f8abad2122a192aa016cc821be95`.
  Content diff confirms the Bots changes are not on main. PR #108 (morning-note
  home) and #111 (delivery dashboard) remain open and are not included.
- This handoff is a subsequent documentation-only commit. Recheck CI on the
  current PR head before merging; the green application commit is not proof of
  green checks on a later revision.
- The Mini's running revision, checkout, launchd labels and UI are **unverified**
  from this machine. Do not treat the development screenshot as a live deployment.

Verification commands used: `git fetch origin`, `git diff --stat origin/main HEAD`,
`gh pr view 112 --json state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup`,
`gh pr checks 112`, `gh run view 38610692902 --log`, and `gh pr list --state open`.

## What is being deployed

Default entry is `#/bots`: named bot cards with purpose, status and latest result.
Open one to send a task through the existing agent API. Drafts and failed-task
messages survive switching bots within the browser session. Work and Projects
remain primary navigation; Today, Inbox, Jobs and detailed views are under More.
Old deep links still work. No bots are created or run merely by visiting a page.

Repository/GitHub prompt collection moves off the event loop, preserving the run
lock. Local `/v1/chat` exposes optional Ollama duration measurements. This is a
responsiveness fix and instrumentation, not a measured inference speedup.

## Deployment sequence

1. Discover the **actual** service checkout and process manager on the Mini.
   Inspect launchd definitions and process command lines locally without printing
   secrets. Record current revision, branch, dirty files, service labels, dashboard
   path and port owners. `git status --short` must be understood before pulling;
   do not reset or stash someone else's work automatically. Record rollback paths
   and preserve the current dashboard build using the established local backup
   procedure. Coordinate with any active auto-deploy job before changing releases.

2. In a clean review checkout, fetch and inspect the current PR:

   ```sh
   git fetch origin
   gh pr view 112 --json state,isDraft,headRefOid,mergeable,reviewDecision
   gh pr checks 112
   ```

   Proceed only when the current head is green and repository review requirements
   are met. Mark #112 ready and merge through the normal repository process. Do
   not bypass branch protections or merge #108/#111. If the PR gained application
   changes since the verified commit, review those changes before proceeding.

3. Update the confirmed clean live checkout to merged main using a fast-forward
   update. Use its existing Python environment/install procedure. Build the
   dashboard explicitly, unless the inspected deployment hook already does so:

   ```sh
   cd dashboard
   npm ci
   npm run lint
   npm test
   npm run build
   ```

   These commands passed in GitHub CI; they have not yet been run on this Mini for
   this release. Do not run another build concurrently with browser tests.
   `scripts/mini-deploy.sh` only pulls, installs Python and optionally invokes
   `AIIA_POST_DEPLOY_HOOK`; it does not itself build the frontend. It also exits
   early if Git SHAs match, so a previous failed build/restart may need explicit
   repair even with an up-to-date checkout.

4. Restart the confirmed Brain and Command Center services through their existing
   manager. Do not guess a launchd label from old docs. Verify the loaded frontend
   directory: Command Center prefers the checkout's `dashboard/dist`, then
   `AIIA_DASHBOARD_DIST`, then a legacy path. Reinstalling Python alone is insufficient.

## Live acceptance checks

```sh
git rev-parse HEAD
lsof -nP -iTCP:8100 -sTCP:LISTEN
lsof -nP -iTCP:8200 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8100/health | jq '.airgap'
curl -fsS http://127.0.0.1:8200/api/health
curl -fsS http://127.0.0.1:8200/api/voice/status | jq '{status, configured, reason}'
curl -fsS http://127.0.0.1:8200/ >/dev/null
```

Confirm airgap is enabled and egress matches the pre-deploy baseline. Confirm port
owners run the intended checkout/environment. A historical packaged Brain process
has previously occupied 8100 while the repository service failed to start; the
tell is a healthy endpoint serving old behavior despite a successful Git update.
Diagnose ownership; do not kill unrelated processes blindly.

Open the authenticated Studio in a browser at a URL **without a hash**: it must
land at `#/bots`. An existing `#/overview` bookmark deliberately remains Overview.
Verify the real roster, open a bot, switch between bots with an unsent draft, and
check Work, Projects, More and browser Back. Check desktop and narrow mobile width;
selecting a lower bot card must reveal its task panel without opening the keyboard.
Attach screenshots; redact sensitive output before sharing.

For live inference, use one existing tool-free bot only, with a harmless synthetic
prompt such as “Reply with exactly: Studio smoke test complete.” Check its configured
tools before sending; do not run a client delivery agent as a smoke test. If no
tool-free bot exists, report that inference verification remains outstanding rather
than changing agents or starting real work. Confirm one request produces one run
and its result appears. UI tests in CI used synthetic records, so this Mini check
is the first real-host verification for this slice.

## Rollback and reporting

If health or the UI regresses, stop further task submissions. Use the recorded
prior release/build and existing service rollback procedure, coordinating with
auto-deploy so it cannot immediately overwrite the rollback. Restore code/build
only, leaving runtime data and credentials intact. Do not use `git reset --hard`;
if no release rollback exists, prepare the previous revision in a separate
checkout and confirm service paths before switching. Re-run health and port-owner
checks after rollback. No schema migration is introduced by this PR.

Return: merged PR and commit, deployed commit, actual service paths/labels, frontend
build result, health/airgap result, browser screenshots, smoke-run outcome and any
remaining issue. “Git pulled” alone is not a deployment result.

Optional after deployment: `python3 scripts/benchmark_studio_models.py` inventories
Mini Ollama without inference. Record hardware and inventory. Do not replace models
or claim speed gains yet; see [model evaluation notes](STUDIO-BOTS-AND-LOCAL-MODELS.md).

## Fresh eyes ledger

1. Application commit and all six CI checks → VERIFIED against GitHub this run.
2. Mini checkout, process labels and current release → UNOWNED until inspected there.
3. Pull/install can succeed without a frontend build or restart → LANDMINE; explicit checks above.
4. Existing Overview bookmarks still open Overview → LANDMINE; test a hash-free URL.
5. #108 chooses a different default home → DECIDE; do not merge it during this deploy.
6. Synthetic screenshots and MacBook model inventory are not Mini evidence → LANDMINE; verify locally.
