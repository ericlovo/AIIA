# Mac Mini official update path

The production Brain checkout on the Mac Mini is `~/aiia-brain/AIIA-public`
on `main`. Command Center's built-in `repo_sync` task (every 6 hours) is the
**official update path** for that checkout. Merging to GitHub does not deploy
the Mini. Green CI does not deploy the Mini.

## Default: fetch-only

`repo_sync` runs `git fetch origin`, then:

1. Records the checkout's HEAD **before** any further git work.
2. Reports how far that HEAD is behind `origin/main` (`git rev-list --count HEAD..origin/main`).
3. Re-indexes files that changed between the last indexed SHA and **current HEAD**.
4. Leaves HEAD where it is. Fetch-only must not fast-forward, merge, or reset.

The task summary reports real HEAD movement this run (before SHA → after SHA,
or "did not move"). The stored `last_commit_sha` is only the last *indexed*
HEAD. A stale stored SHA is not a deploy gap: if HEAD is already at
`origin/main`, the summary says HEAD did not move even when indexing catches up.

## Opt-in pull

Set `AIIA_REPO_SYNC_PULL=1` in the Command Center / Brain environment to allow
a fast-forward of `origin/main`. Anything other than `1` stays fetch-only.

When pull is enabled, `repo_sync`:

1. Refuses if the tree has **tracked** changes (`git status --porcelain --untracked-files=no`). Untracked runtime files do not block a pull.
2. Refuses if `git pull --ff-only origin main` cannot fast-forward.
3. After a successful pull, GETs Brain `/health` on port 8100 (`AIIA_URL`, default `http://localhost:8100`).
4. Writes the old SHA, the new SHA, and the verify result into the task summary.

A failed `/health` verify **fails the task**. `repo_sync` does not restart
launchd, kickstart Brain, or otherwise auto-recover. Fix or restart by hand,
then confirm `curl -fsS http://127.0.0.1:8100/health`.

## What this is not

- It is not a merge-to-live hook. Default `repo_sync` does not move HEAD.
- It does not install Python packages, build `dashboard/dist`, or rewrite
  launchd plists.
- It does not treat GitHub merges, green Actions, or a stored SHA gap as proof
  that the Mini is running the new revision.
- `scripts/mini-deploy.sh` is a separate LaunchAgent helper (fetch, optional
  pull, `pip install`, optional `AIIA_POST_DEPLOY_HOOK`). It is not the
  Command Center scheduler. Do not run it as a silent deploy alongside
  fetch-only `repo_sync` without the same dirty/ff and `/health` checks.

## After an intentional pull

1. Confirm HEAD: `git -C ~/aiia-brain/AIIA-public rev-parse --short HEAD`
2. Confirm the listener is the repo Brain, not a packaged binary:
   `lsof -nP -iTCP:8100 -sTCP:LISTEN`
3. Confirm health: `curl -fsS http://127.0.0.1:8100/health`
4. Restart Brain / Command Center yourself if the new revision must be loaded
   into the running processes. `repo_sync` will not do that.
