# Ignored-rule ratchet

Run `python scripts/ruff_ratchet.py` after installing `.[dev]`. CI runs it in
lint-and-test alongside ordinary Ruff lint, formatting and pytest. The initial
baseline is 101 violations across 13 ignored rules, measured using Ruff 0.16.9.

The script reads pyproject.toml and scripts/ruff-baseline.json, explicitly enables
the backlog rules, and compares JSON statistics per rule. E501 (formatter-owned
line length policy) and B008 (FastAPI defaults) remain intentional exemptions.
Each globally ignored rule other than those two must have a baseline entry.

Exit 0 means every count is at or below its ceiling. Exit 1 means at least one
ceiling increased. Exit 2 means invalid config/baseline, Ruff failure or malformed
statistics; those conditions never pass silently. Ruff runs through the same
Python environment as the script, without autofixes. Python 3.10 uses the
conditional tomli development dependency; Python 3.11+ uses tomllib.

## Cleanup workflow

Fix one rule per PR, run tests, and lower its baseline count. At zero, remove the
rule from both ignore and the baseline, so ordinary lint enforces it thereafter.
Do not regenerate ceilings automatically to make CI green. Tool upgrades that
change diagnostics need review rather than an automatic baseline increase.

This is a count ceiling, not a per-line ledger: removing one violation can offset
adding another under the same rule. A reduced count does not automatically lower
the ceiling; reviewers must do that in cleanup PRs. Per-file ignores, noqa and
Ruff file exclusions remain effective and need ordinary code review. It is not
an authorization mechanism against someone editing CI or its baseline.

Tests include an actual Ruff invocation where ordinary lint passes two ignored
B006 findings but the ratchet fails against a ceiling of one, plus malformed
output, tool failure, baseline drift and boundary cases. No backlog cleanup or
runtime behavior changes are included in this PR.
