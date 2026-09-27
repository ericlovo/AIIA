# Bugbot rules for AIIA

Bugbot is billed per run from included usage, so it is run on request
(`bugbot run`) for high-risk PRs, not on every push. See
`docs/REVIEW-BOTS.md` for who reviews what.

## Report only

- Correctness bugs with a concrete failure scenario (inputs → wrong result).
- Security defects: auth bypass, secrets in logs or responses, injection,
  unsafe deserialization, path traversal.
- Broken invariants listed below.

## Do not report

- Style, formatting, naming, or anything `ruff` or `eslint` enforces.
- Missing docstrings or comments.
- Suggestions without a failure scenario.

## Invariants

- `local_brain/execution/`: an action must pass the pre-execution gate in
  `safety.py`. No path may skip it, downgrade a tier (AUTO / SUPERVISED /
  GATED), widen an allowlist, or add a fourth executor path.
- `local_brain/egress.py`: under `AIIA_AIRGAP`, deny except the explicit
  allowlist. With Sanction configured, every failure mode denies. With
  Sanction unconfigured, egress is allowed by design; that is not a bug.
- `local_brain/command_center/`: agent-to-agent work moves only through a
  typed Handoff (`docs/EXECUTABLE-ORGANIZATION.md`). Human approvals
  (actions, git workspaces, git writes) must never be granted by code.
- Runtime JSON (`action_data.json`, `agent_data.json`, `task_data.json`)
  is never committed.

## Ignore

`dashboard/dist/`, `**/node_modules/`, `**/package-lock.json`,
`docs/eval-results/`.
