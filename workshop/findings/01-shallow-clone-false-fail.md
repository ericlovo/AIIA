# Finding 01 — `archive-dogfood` reports a false FAIL on a shallow clone

**Severity:** low (no bad code ships) / **high friction** (it's the first thing a newcomer sees)
**Where:** `scripts/eval-archive-dogfood.py`, surfaced via `scripts/eval.sh --check=archive-dogfood`
**Status:** confirmed, unfixed

## What happens

Clone Writ the way most people first clone anything, and the harness fails:

```bash
git clone --depth 1 https://github.com/sellke/writ /tmp/writ
cd /tmp/writ && bash scripts/eval.sh
```

```
## archive-dogfood
FAIL (1 finding(s))
- `archive-dogfood:dogfood-commit-recorded-as-renames`:
   _Remediation:_ Investigate the real .writ/specs/archive/ tree —
   this check runs against production data, not a fixture.
Scenarios: 14/15 passed
```

Everything else is green. 44 of 45 checks pass; this one doesn't.

## What's actually wrong

Nothing is wrong with the repo. The check reads real commit history to verify that
spec-archival commits were recorded as git *renames* rather than delete+add. A
`--depth 1` clone has one commit, so the rename evidence isn't there to find.

Unshallow and it passes:

```bash
git -C /tmp/writ fetch --unshallow      # 470 commits
bash scripts/eval.sh --check=archive-dogfood
# PASS — Scenarios: 15/15 passed
```

Verified on this sandbox: shallow → FAIL, full history → 15/15 PASS, same tree.

## Why it matters more than the severity suggests

The remediation string sends you the wrong way. It says *"Investigate the real
`.writ/specs/archive/` tree"* — so you go read archive directories looking for a
data problem that does not exist. The actual fix is one `git fetch --unshallow`,
and nothing in the output mentions clone depth.

This is the harness's own failure mode: a check that reads production git state
can't distinguish "the evidence is absent" from "the evidence is wrong," and the
message it emits assumes the second.

## Proposed fix

Detect the degraded input and say so, rather than reporting a data finding.
`scan_repo_citations` in `ac-trace.py` already models this well — it reports
`scanned_files` and `ignore_filter` "so a pathological or degraded scan is visible
rather than silently narrowed." `archive-dogfood` should do the same:

```python
if (repo / ".git" / "shallow").exists():
    # A shallow clone has no rename history to inspect. Report the
    # degraded input; do not report a data finding.
    return note("archive-dogfood:shallow-clone",
                "Shallow clone — rename evidence unavailable. "
                "Run `git fetch --unshallow` to enable this check.")
```

That turns a misleading FAIL into an accurate non-blocking NOTE.

## Workshop question

Which other checks read production git state and would degrade the same way?
`post-merge-archival`, `git-notes-audit`, and `supersession-writeback` are the
candidates — they passed here only because full history was present. Is
"detect degraded input" a per-check concern, or should `eval.sh` gate the whole
git-reading class of checks behind one up-front repo-capability probe?

## Mitigation in this sandbox

`workshop/bootstrap.sh` clones full-depth and repairs a pre-existing shallow
clone; `run-harness.sh` prints a warning if it sees `.git/shallow`.
