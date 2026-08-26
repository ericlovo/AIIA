# Writ Harness Workshop Sandbox

Prep for the in-person session on **Writ's internals** — the execution harness
(gates, evals, `scripts/`) and the authoring surface (`/new-command`,
`/new-skill`, `/refresh-command`).

## Ground rule: we go in on his tree, exactly as he built it

We're working in the author's session. The starting state is
[sellke/writ](https://github.com/sellke/writ) `v0.33.0` @ `5b9082d` —
`HEAD == origin/main`, zero diff, nothing applied.

**Nothing in this folder is pre-applied to his checkout.** The one fix we carry
lives as a patch file and stays there until he wants it. Everything here is
read-only against his tree, and the harness runner cleans up after itself.

Prove it before you start:

```bash
bash workshop/bootstrap.sh --verify
```

```
  version:  0.33.0  @ 5b9082d
  modified: 0 tracked file(s)
  stray:    0 ignored path(s)
PRISTINE — this is Writ exactly as shipped.
```

`bootstrap.sh` (no args) clones full-depth if the checkout is missing, then
verifies. `--reset` forces back to pristine and **discards local edits** — only
for your own scratch clone, never his.

## Run the harness

```bash
bash workshop/run-harness.sh              # four layers, ~40s
bash workshop/run-harness.sh eval|tests|measure|parity
WRIT_DIR=/path/to/writ bash workshop/run-harness.sh
```

It prints whether the tree is pristine before it runs, and removes
`__pycache__`, `.pytest_cache`, and any `.writ/state/` it created. Verified:
0 modified, 0 stray after a full run.

## What pristine Writ actually does

This is his shipped state, not a cleaned-up version of it:

```
── [1/4] eval.sh — 45 static checks (what CI runs)
    45 pass / 0 fail
── [2/4] pytest scripts/tests/ — unit suite (CI does NOT run this)
    FAILED test_ac_trace.py::CitationScanTests::test_symlink_loop_does_not_crash_the_scan
    1 failed, 791 passed, 3 skipped
── [3/4] measure-invocation.py — per-command context load
    shared base (every invocation): 26,437 bytes
    commands: 31   floor min/median/max: 32,388 / 40,093 / 75,433
── [4/4] check-agent-parity.sh — cross-platform agent alignment
    parity OK
```

**CI is green and the unit suite is not.** That gap is the whole story, and it
is visible in the first 40 seconds.

## The four layers, and what each protects

| Layer | Command | Count | In CI? |
|---|---|---|---|
| Static checks | `scripts/eval.sh` | 45 checks | **yes** — `.github/workflows/eval.yml` |
| Unit tests | `pytest scripts/tests/` | 792 tests | **no** |
| Context budget | `measure-invocation.py` | 31 commands | no |
| Agent parity | `check-agent-parity.sh` | 3 platforms | no |

The 45 checks aren't generic linting — they enforce the methodology itself:
`anti-sycophancy` greps the Prime Directive's banned phrases, `leanness` ratchets
per-surface byte ceilings against written justifications, `loop-bounds` proves
every autonomous loop terminates, `ac-trace` verifies acceptance criteria are
cited by real tests. Executable methodology is the interesting claim, and it
mostly holds up.

## Findings — observations to bring, not changes to make

| # | Finding | Severity | State |
|---|---|---|---|
| [01](findings/01-shallow-clone-false-fail.md) | `archive-dogfood` false-FAILs on a shallow clone; remediation text misdirects | low / high friction | confirmed |
| [02](findings/02-ac-trace-symlink-loop.md) | `ac-trace.py` crashes on symlink loops (Python ≤3.12); its own test catches it, CI never runs it | medium | fix written + verified, **not applied** |
| [03](findings/03-context-budget.md) | Shared base outweighs the command file in 25 of 31 commands; token ratio never validated | design tension | measured |
| [04](findings/04-knowledge-writeback-corruption.md) | **All 10 phase-close lessons are corrupted.** String `evidence` iterated per character; missing `statement` writes an empty TL;DR and reports success | **high** | fix + 10 tests verified, **not applied**; recovery tool included |

Finding 02's patch is `patches/0001-ac-trace-symlink-loop-runtimeerror.patch`.
If he wants it in the room:

```bash
# absolute path — git -C resolves relative paths against the TARGET repo
git -C "$WRIT_DIR" apply "$PWD/workshop/patches/0001-ac-trace-symlink-loop-runtimeerror.patch"
bash workshop/run-harness.sh tests                   # 792 passed
git -C "$WRIT_DIR" checkout -- scripts/ac-trace.py   # back to pristine
```

Both directions verified on his checkout: pristine → 1 modified → 792 passed →
reverted → 0 modified, 0 diff vs `origin/main`. It's two lines plus comments.

While the patch is applied, `run-harness.sh` prints
`Tree: 1 MODIFIED file(s) — results do NOT reflect upstream`, so a patched run
can't be mistaken for his shipped state.

## Four things worth arguing about

0. **The learning loop is writing garbage, and has been since day one.** All 10
   automated lessons are malformed (finding 04) — including, with some irony,
   the one recording what progressive disclosure costs. `phase-knowledge` and
   `knowledge-consolidate` are green on that data. Lead with this one.

1. **The CI gap.** 792 tests with nothing in front of them. The leanness ledger
   defends this deliberately — eval scenarios plus `require_literal` bindings are
   "each checker's entire CI protection." Findings 02 and 04 are the second and
   third documented escapes; the ledger records the first itself. Does the model
   hold, or has it now failed three times?

2. **Progressive disclosure is running a measured loss.** Skills extraction
   bought a −35.9% floor and cost a +9.7% ceiling, at ~1,017 bytes of overhead
   per extracted skill. Meanwhile the 26KB shared base dominates 25 of 31
   invocations and nobody has cut it. The effort may be aimed at the wrong number.

3. **The token budget is an unvalidated guess.** `chars/4`, inherited from the
   roadmap, never checked against a tokenizer — the script says so itself. Every
   leanness ceiling and ADR-021 decision inherits that. Calibrating it is small
   work that re-prices everything else.

These are his tradeoffs, made on purpose and documented honestly — the leanness
ledger openly refuses to run `--update-baseline` because it "would erase the
commands justifications." Go in curious about why, not with a verdict.

## Layout

```
workshop/
├── bootstrap.sh       clone full-depth + verify pristine (--verify, --reset)
├── run-harness.sh     run four layers read-only, results to baseline/
├── findings/          four findings with repro steps
├── patches/           0001 ac-trace symlink fix        } both verified,
│                      0002 knowledge-writeback + tests } neither applied
├── tools/
│   └── recover-lessons.py   repair the 10 corrupted lessons (dry-run default)
└── baseline/          generated: eval.md, tests.txt, invocation.txt
                       (captured from the pristine tree)
```

## Separately: Writ is installed in this repo

AIIA has Writ v0.33.0 installed for practice — 31 commands, 6 subagents, 16
skills, `.writ/` workspace. Try `/status` or `/create-spec` here, not in his tree.

Two things to know:

- The install put 21 scripts into AIIA's top-level `scripts/` alongside AIIA's
  own. Nothing was overwritten (verified), but namespaces are mixed. Writ's
  commands hardcode `scripts/<name>.py`, so moving them breaks the commands.
- `CLAUDE.md` gained a Writ block between `<!-- writ:start -->` / `<!-- writ:end -->`;
  AIIA's original content sits above it.

`/uninstall-writ` removes the platform files and keeps `.writ/`.
