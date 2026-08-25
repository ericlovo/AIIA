# Writ Harness Workshop Sandbox

A working sandbox for the in-person session on **Writ's internals** — its
execution harness (the gates, evals, and `scripts/`) and its authoring surface
(`/new-command`, `/new-skill`, `/refresh-command`).

Not a demo of an empty install. Everything here has been run against
[sellke/writ](https://github.com/sellke/writ) `v0.33.0` @ `5b9082d`, and the
findings are real.

## Start here

```bash
bash workshop/bootstrap.sh      # clone Writ full-depth + install pytest
bash workshop/run-harness.sh    # run all four layers, ~40s
```

Expected on a healthy checkout:

```
── [1/4] eval.sh — 45 static checks (what CI runs)
    45 pass / 0 fail
── [2/4] pytest scripts/tests/ — unit suite (CI does NOT run this)
    792 passed, 3 skipped in 15.93s
── [3/4] measure-invocation.py — per-command context load
    shared base (every invocation): 26,437 bytes
    commands: 31   floor min/median/max: 32,388 / 40,093 / 75,433
── [4/4] check-agent-parity.sh — cross-platform agent alignment
    parity OK
All green.
```

Note the "792 passed" line is only green *because* of `patches/0001` — see
finding 02. On a clean upstream checkout it reads `1 failed, 791 passed`.

Individual layers: `run-harness.sh eval|tests|measure|parity`.
Point at a different checkout with `WRIT_DIR=/path/to/writ`.

## The four layers, and what each one actually protects

| Layer | Command | Count | Wired into CI? |
|---|---|---|---|
| Static checks | `scripts/eval.sh` | 45 checks | **yes** — `.github/workflows/eval.yml` |
| Unit tests | `pytest scripts/tests/` | 792 tests | **no** |
| Context budget | `measure-invocation.py` | 31 commands | no |
| Agent parity | `check-agent-parity.sh` | 3 platforms | no |

That second row is the single most important fact about this harness, and it's
where finding 02 came from.

The 45 checks are not generic linting. They enforce Writ's own methodology:
`anti-sycophancy` greps the Prime Directive's banned phrases, `leanness` ratchets
byte ceilings per surface with written justifications, `loop-bounds` proves every
autonomous loop terminates, `ac-trace` verifies acceptance criteria are cited by
real tests. The methodology is executable, which is the interesting claim.

## Findings

| # | Finding | Severity | Status |
|---|---|---|---|
| [01](findings/01-shallow-clone-false-fail.md) | `archive-dogfood` false-FAILs on a shallow clone, with misdirecting remediation | low / high friction | confirmed, unfixed |
| [02](findings/02-ac-trace-symlink-loop.md) | `ac-trace.py` crashes on symlink loops (Python ≤3.12); its own test catches it, CI never runs it | medium | **fix verified**, `patches/0001` |
| [03](findings/03-context-budget.md) | Shared base outweighs the command file in 25 of 31 commands; token ratio never validated | design tension | measured |

### Applying the fix

```bash
git -C "${WRIT_DIR:-$HOME/sellke/writ}" apply \
    workshop/patches/0001-ac-trace-symlink-loop-runtimeerror.patch
bash workshop/run-harness.sh tests     # 792 passed
```

This session can't push to `sellke/writ` (cross-owner attach is blocked), so the
fix travels as a patch. It's a two-line change plus comments — ready to be a PR
from a machine with push rights.

## Three things worth arguing about in the room

1. **The CI gap.** 792 tests with nothing in front of them. The leanness ledger
   defends this as deliberate — eval scenarios plus `require_literal` bindings are
   "each checker's entire CI protection." Finding 02 is the second documented
   escape through that gap; the ledger itself records the first. Does the model
   hold, or has it now failed twice?

2. **Progressive disclosure is running a measured loss.** Skills extraction bought
   a −35.9% floor and cost a +9.7% ceiling, at 1,017 bytes of overhead per
   extracted skill. Meanwhile the 26KB shared base dominates 25 of 31 invocations
   and nobody has cut it. The effort may be aimed at the wrong number.

3. **The whole token budget is an unvalidated guess.** `chars/4`, inherited from
   the roadmap, never checked against a tokenizer. Every leanness ceiling and every
   ADR-021 decision inherits that. Calibrating it is a small piece of work that
   re-prices every other decision.

## Layout

```
workshop/
├── bootstrap.sh       clone Writ full-depth, install pytest
├── run-harness.sh     run the four layers, write results to baseline/
├── findings/          the three findings, with repro steps
├── patches/           0001 — the verified ac-trace fix
└── baseline/          generated: eval.md, tests.txt, invocation.txt
```

## Writ is also installed in this repo

Separately from the sandbox, Writ v0.33.0 is installed into AIIA itself —
31 commands in `.claude/commands/`, 6 subagents in `.claude/agents/`, 16 skills
in `.claude/skills/`, workspace in `.writ/`. Try `/status` or `/create-spec`.

Two things to know about that install:

- **It dropped 21 scripts into AIIA's top-level `scripts/`**, alongside AIIA's own.
  Nothing was overwritten (verified), but the namespaces are now mixed. Writ's
  commands hardcode `scripts/<name>.py`, so moving them breaks the commands.
- `CLAUDE.md` gained a Writ block between `<!-- writ:start -->` / `<!-- writ:end -->`
  markers; AIIA's original content is preserved above it.

`/uninstall-writ` removes the platform files and keeps `.writ/`.
