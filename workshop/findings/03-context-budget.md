# Finding 03 — the shared base outweighs the command in 25 of 31 commands

**Severity:** not a bug — a design tension, and the best argument in the room
**Where:** `scripts/measure-invocation.py`, ADR-021 (progressive disclosure token budget)
**Status:** measured, unresolved by design

## The measurement

```
shared base (every invocation): 26,437 bytes
    system-instructions.md               20,779
    commands/_preamble.md                 5,658
```

Every single `/command` pays that 26KB before its own file is read. Against
per-command floors:

| | command | floor | base% |
|---|---|---|---|
| heaviest | `create-spec` | 75,433 | 35.0% |
| median | `migrate` | 40,093 | 65.9% |
| lightest | `create-issue` | 32,388 | **81.6%** |

**25 of 31 commands are majority shared base.** For `/create-issue` — a command
whose whole pitch is "under 2 minutes" — 82% of what loads is boilerplate it
never uses. Only six commands (`create-spec`, `implement-phase`, `verify-spec`,
`release`, `ship`, `implement-story`) carry more of their own weight than the
preamble's.

Worst full path: `implement-story` at **105,717 bytes** (54,127 floor + 51,590
conditional).

## The tension ADR-021 already found

Writ's own leanness ledger records the result of the skills-extraction pilot,
and it is not a clean win:

> "extraction bought a **-35.9% floor** on this command and cost a **+9.7% full-path
> ceiling**, and the overhead is the whole cost."

> "progressive disclosure costs **1,017 bytes of overhead per extracted skill**"
> — `.writ/knowledge/lessons/2026-08-12-*`

So pulling prose out of `implement-story` into skills made the common path
cheaper and the complete path more expensive. Sixteen skills × ~1KB of scaffolding
is ~16KB of pure overhead bought to move text around.

The ledger is honest about this: `skills.chars` currently sits at **88,203**,
past its justified ceiling of 78,719 — a `WARNING`, deliberately not silenced.
The note says `--update-baseline` "was deliberately NOT run: it would move every
surface's floor and erase the commands justifications."

That is unusually disciplined instrumentation. It's also an unresolved problem
being carefully watched rather than fixed.

## The measurement's own caveat

`measure-invocation.py` says this out loud, which is the right instinct:

> "Bytes are measured. Tokens are NOT measured: no tokenizer was available... The
> chars/4 ratio inherited from `.writ/product/roadmap.md` has **never been
> validated against a real tokenizer** — treat every `*_tokens_estimated` value as
> an order-of-magnitude figure."

So the entire token budget — the thing ADR-021 governs, the thing the leanness
ceilings ratchet against — rests on an unvalidated 4:1 guess. Every budget
decision made so far is an order-of-magnitude decision.

Anthropic's tokenizer is reachable via the count-tokens endpoint. Calibrating
`--chars-per-token` once would convert this whole apparatus from estimated to
measured, and it is a genuinely small piece of work.

## Workshop questions

1. **Is the shared base the actual target?** Cutting `system-instructions.md`
   by 30% saves ~6KB on *every* invocation — more than any single command
   extraction has returned. Why has the effort gone to per-command extraction
   instead of the constant that dominates 25 of 31 loads?
2. **Does the 1,017-byte-per-skill overhead justify more extraction?** The pilot's
   stated input for the remaining five specs is "fewer, larger skills." Does that
   survive contact with the measurement, or does it just make the ceiling worse
   more slowly?
3. **Calibrate the ratio first.** Every answer above changes if the real ratio is
   3:1 or 5:1. This should probably happen before the next extraction spec.

## Reproduce

```bash
bash workshop/run-harness.sh measure
cat workshop/baseline/invocation.txt
```
