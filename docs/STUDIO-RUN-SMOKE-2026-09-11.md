# Studio RUN Smoke Slice: 2026-09-11

Status: **DEPLOYED and SMOKE PASSED (2026-09-11 16:14–16:30 CDT).** Deploy commands were run by the owner from the terminal; verification and the smoke were run by Claude Code.

Candidate frozen at `f9e7b7e2caffc95aa4e57f9ee33655dac1e5d37a` on
`codex/studio-mini-reconcile-20260911` = `ef7d3cf` + package patch 01
(`b78985e`) + patch 02 (`ad5e95e`) + reliability delta. Not pushed.

## Installed build inventory (read-only, 2026-09-11 15:30 CDT)

| Item | Observed |
| --- | --- |
| Checkout | `~/aiia-brain/AIIA-public`, `main` at `fdbf863`, two commits behind remote main (`ef7d3cf`) |
| Dirty files | `local_brain/egress.py`, `local_brain/command_center/voice_conductor.py` (older local xai.realtime allowlist; behaviorally superseded by merged #53), `monitor_data.json` (runtime) |
| Service | launchd `gui/501/com.aiia.brain` → `~/aiia-brain/start_aiia.sh`; `AIIA_AIRGAP=1`; KeepAlive; `com.aiia.brain-watchdog` kickstarts on 8100 unhealthy |
| Processes | launcher 47752; Brain 47758 (`local_brain.local_api:app`, 127.0.0.1:8100); Command Center 47763 (`local_brain.command_center.server:app`, 127.0.0.1:8200); started 2026-09-08 09:57:53; cwd `AIIA-public`; `.venv` Python 3.12 |
| Health | 8100 `/health` airgap enabled, `xai.realtime` allowlisted, all other egress disabled; 8200 voice status `connected`, configured, no reason |
| Served dashboard | `assets/index-C_Zaic-W.js` + `index-C0wHiHKD.css`, built 2026-09-07 15:38, SHA-256 identical to `dashboard/dist`; bundle contains no `Switchboard`, `studio/activity`, or RUN-button strings; `GET /api/studio/activity` → 404 |
| Model | `LOCAL_TASK_MODEL=qwen3:8b` (also routing/deep); present in `ollama list` |
| Runtime | 23 agents, 2 loop-enabled (Mindmoor Delivery Watch 8/8, CI Signal Officer 12/12 daily budget used); 48 assignments (46 completed, 2 failed); 6 handoffs; no SQLite ledger yet |

Conclusion: the blue RUN button and Switchboard are absent from the installed
build because the feature was never deployed, not because of a UI defect.

## Patch comparison

Applied `01-studio-integration.patch` then `02-run-buttons.patch` on `ef7d3cf`
in a temporary worktree and diffed the result against the reconcile tree
(excluding venv, node_modules, dist, runtime JSON). Only these differ:

- `local_brain/command_center/{agent_registry,assignment_registry,server}.py` (reliability fixes)
- `local_brain/tests/test_agent_reliability.py`, `test_assignment_reliability.py` (new)
- `docs/STUDIO-RECONCILIATION-2026-09-11.md`, `docs/STUDIO-RELEASE-GATE-2026-09-11.md` (new)
- `dashboard/src/console/AgentStudio.tsx` (+2 lines), `CHANGELOG.md`

Everything else in the reconcile tree is byte-identical to the patches.

## Migration dry-run

Copies of the live `agent_data.json` and `assignment_data.json` loaded with the
candidate registries in a temp dir: 23 agents, both loops, statuses all idle,
48 assignments, 6 handoffs. Re-save changed no existing field; it only added
agent `suite`, `memory_namespace`, top-level `pending_runs`, and assignment
`review_status`, `review_note`, `reviewed_at`, `review_version`, plus a new
`agent_data.runs.sqlite3`.

## Backup

`~/aiia-local-backups/studio-deploy-2026-09-11/`: all `command_center/*.json`
(parsed OK, hashes match live), served `dashboard-dist/`, production local
edits as `production-local-edits.patch`, `INVENTORY.txt`. Taken with all agents
idle. Retake if the deploy happens on a later day.

## Deploy plan (executed by owner 2026-09-11)

Prerequisite: production dashboard lacks `lucide-react`, which the candidate adds.

```sh
cd ~/aiia-brain/AIIA-public
git stash push -m "pre-deploy 2026-09-11: local airgap/voice edits superseded by #53" -- local_brain/egress.py local_brain/command_center/voice_conductor.py
git checkout --detach f9e7b7e2caffc95aa4e57f9ee33655dac1e5d37a
cd dashboard && npm install --no-audit --no-fund && npm run build && cd ..
launchctl kickstart -k gui/501/com.aiia.brain
```

Verify after restart:

```sh
lsof -nP -iTCP:8100 -iTCP:8200 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8100/health | jq '.airgap.enabled, .airgap.egress["xai.realtime"]'
curl -fsS http://127.0.0.1:8200/api/voice/status | jq '{status,configured,reason}'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8200/api/studio/activity
curl -s http://127.0.0.1:8200/api/agents | jq '[.agents|length, [.[]|select(.loop_enabled)|.name]]'
curl -s http://127.0.0.1:8200/api/assignments | jq '.assignments|length'
```

Expect: Python listeners on both ports (not the Console.app bundled binary),
airgap true with `xai.realtime` = `airgap-allowlisted`, voice connected, activity
200, 23 agents with the same two loops enabled, 48 assignments.

Rollback:

```sh
cd ~/aiia-brain/AIIA-public
git checkout main && git stash pop
cp ~/aiia-local-backups/studio-deploy-2026-09-11/command_center/*.json local_brain/command_center/
rm -rf dashboard/dist && cp -R ~/aiia-local-backups/studio-deploy-2026-09-11/dashboard-dist dashboard/dist
launchctl kickstart -k gui/501/com.aiia.brain
```

Preserve `pending_runs` from the new `agent_data.json` before overwriting it if
any run completed between deploy and rollback.

## Smoke (after deploy, at most two inference calls)

1. `POST /api/agents`: name `Test - RUN button smoke`, analysis skill, `tools=[]`,
   no repository, loops off, temperature 0.1, max_tokens 1000. Reload, confirm ID.
2. Seed: one manual run with task
   `Synthetic smoke test: reply with exactly STUDIO_RUN_OK. No tools are needed.`
3. In the dashboard Overview → Activity, click that row's blue RUN once. Observe
   pending state, disabled RUN controls, one new result, one new ledger record.
4. Record HTTP status, model from the response or UNVERIFIED, latency, exact
   task, result, served asset hash. Reload; confirm both records remain.

## Deploy result

- Extra step needed: an untracked local `docs/HANDOFF-ASTRA-STUDIO-TUNEUP-2026-09-07.md`
  (57 lines, different from the tracked 200-line version) blocked checkout; it was
  moved to the backup folder as `...local.md`, not deleted.
- Production checkout now detached at `f9e7b7e`; local airgap/voice edits are in
  `git stash` ("pre-deploy 2026-09-11"); `monitor_data.json` still dirty (runtime).
- `npm install` added `lucide-react`; build produced `index-B03Wt8bk.js` /
  `index-B24L8CgX.css`, identical hashes to the reconcile-worktree build.
- After `launchctl kickstart -k`: Brain PID 61160 on 8100, Command Center PID
  61167 on 8200 (Python, cwd `AIIA-public`, started 16:14:57). Health online,
  airgap enabled, `xai.realtime` = `airgap-allowlisted`, voice connected.
  `/api/studio/activity` 200 with 116 imported legacy runs. 23 agents, both
  loops still enabled and idle, 48 assignments, `agent_data.runs.sqlite3` created.
  Watchdog did not fire.

## Smoke result (two inference calls, both qwen3:8b)

| Step | Evidence |
| --- | --- |
| Agent | `Test - RUN button smoke`, id `9da3d9a90d0f`, skills `[analysis]`, `tools=[]`, no repo, loops off, temperature 0.1, max_tokens 1000; re-fetched from API and present on disk |
| Seed run | `POST /api/agents/9da3d9a90d0f/run` → 200, model `qwen3:8b`, 23,019 ms, result `STUDIO_RUN_OK`, ledger run `1833f0f39ad44e2f8a4bde19de354c1d` trigger `manual` |
| RUN click | Overview → Activity ledger row → one click on the blue RUN. Browser recorded exactly one request: `POST /api/agents/9da3d9a90d0f/run` → 200 |
| Pending feedback | `role=status` region "Running Test - RUN button smoke..." rendered immediately after the click |
| Settled feedback | Sticky banner "Test - RUN button smoke: run completed."; second row "completed a manual run · STUDIO_RUN_OK"; Agent load shows 2 complete |
| New attempt | Ledger run `90b2228f78d94cef9abbeabb368f98a0`, trigger `manual`, 19,991 ms, task byte-identical to the seed, result `STUDIO_RUN_OK`; SQLite has 2 rows for the agent |
| Reload | Full page reload → Overview: both rows present, ledger total 118, agent idle, `pending_runs` empty |
| Fleet | 24 agents (23 + smoke), none running, both loops still enabled; no other agent touched |

Not directly captured: the `disabled` attribute on the RUN buttons during the
pending window (the accessibility read does not expose it, and the two-call
budget did not allow a second timed run). Prior synthetic browser evidence
(four clicks → four POSTs, two 409s) covers busy protection; it was not rerun.

Remaining after this slice: accept/reject/reopen review checks on existing
synthetic assignments, then the frozen six-call baseline and FLOW-01. The seven
`test_streaming_chat.py` live-server tests still need an opt-in gate.

## Review flow verification (same day, no inference)

Target: synthetic assignment `asg_de1130dad895` ("Eval FLOW-01 evidence
audit", Test - Evidence Auditor). Deployed revision `f9e7b7e` on the Mini.

| Step | Path | Result |
| --- | --- | --- |
| Accept with current version | API | 200, `review_status=accepted`, note saved, `reviewed_at` set, version rotated |
| Replay with the stale version | API | 409 `review_changed_refresh_required`, state unchanged |
| Reject with fresh version | API | 200, `rejected`, on disk immediately |
| Review a failed assignment | API | 409 `assignment_not_reviewable` |
| Review unknown id | API | 404 `assignment_not_found` |
| Reopen review | UI button | "Rejected output" label cleared on card and detail panel |
| Accept output | UI button | "Accepted output" label plus "Decision saved" timestamp |
| Full page reload | UI + API + disk | Card still "Accepted output"; API and `assignment_data.json` agree; Switchboard attention count 48 → 47 |

Observation, not a defect against the contract: the review note textarea keeps
the previous decision's text, so accepting after a reject carried the "Rejected
during…" note forward until it was edited. Consider clearing the draft note on
Reopen, or labeling the note with the decision it was written for.

Tooling note: an earlier API pass looked like a JSON error; it was zsh `echo`
expanding `\n` inside the response, not the server. Raw bodies are valid JSON.
