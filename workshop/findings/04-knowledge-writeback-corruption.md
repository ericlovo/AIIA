# Finding 04 — the learning loop has never once written a valid artifact

**Severity:** high — silent data loss in the mechanism Writ uses to learn
**Where:** `scripts/phase-state.py` `knowledge_writeback()` (≈ lines 782–850)
**Status:** confirmed and reproduced. Fix + 10 tests written and verified —
**not applied**. Recovery tool included.

## What's wrong

All **10** phase-close lessons in `.writ/knowledge/lessons/` are malformed.
Every one written by the automated writeback path. The 6 hand-authored lessons
(2026-04-24, 2026-07-11, 2026-07-19) are fine.

From `2026-08-12-progressive-disclosure-costs-1-017-bytes-of-overhead-per-extracted-skill.md`:

```yaml
related_artifacts:
  - .
  - /
  - /
  - /
  - .
```

```markdown
## TL;DR


## Context

**Cited evidence:**

- .
- w
- r
- i
- t
- /
- s
...
```

Two independent defects, both silent:

1. **A string `evidence` is iterated per character.** `related_artifacts` fills
   with `.` and `/`; the cited-evidence list becomes one bullet per letter.
2. **A missing `statement` writes an empty `## TL;DR`** and reports success.

All 10 files have both. The headline of every lesson — the thing the loop exists
to capture — is blank.

## Root cause

```python
statement = cand.get("statement", "")     # never checked for emptiness
...
evidence = cand.get("evidence", [])
artifact_paths = cand.get("artifacts") or [e for e in evidence if _looks_like_path(e)]
context += "\n\n**Cited evidence:**\n\n" + "\n".join(f"- {e}" for e in evidence)
```

The code is correct *if* `evidence` is a list. A `str` is also iterable, so
`for e in evidence` walks characters. The guard on line 810 —
`if not cand.get("evidence")` — passes a non-empty string happily.

The punctuation survives into `related_artifacts` because the path heuristic
accepts it:

```python
def _looks_like_path(token):
    t = token.strip()
    if not t or " " in t: return False
    return "/" in t or "." in t     # "." -> True,  "/" -> True
```

## Reproduce (30 seconds, read-only)

```bash
cat > /tmp/c.json <<'EOF'
{"candidates":[{"id":"L1","title":"t","statement":"s","generalizes":true,
 "evidence":".writ/specs/x/load-report.md; measured 2026-08-12"}]}
EOF
python3 scripts/phase-state.py knowledge-writeback --candidates /tmp/c.json --knowledge-dir /tmp/k
cat /tmp/k/lessons/*.md
```

Byte-for-byte reproduction of the damage in his repo. Passing the same value as
a **list** renders correctly — that's the whole difference.

Omitting `statement` entirely returns
`{"written": [...], "rejected": []}` — exit 0, success reported, empty TL;DR on disk.

## Why nothing caught it

- **No unit tests.** `test_phase_state.py` has 19 tests; none reach
  `knowledge_writeback`. It is the only writer of `.writ/knowledge/lessons/`.
- **The eval checks pass on the corrupted data.** `phase-knowledge` and
  `knowledge-consolidate` are green in his repo *right now*, with all 10 files
  malformed. They assert the writeback ran and that references resolve — not
  that the artifact says anything.
- **`.` and `/` resolve as paths**, so the "dangling reference" detector that
  might have flagged them sees valid repo paths.

This is finding 02's pattern again, one layer up: the gate checks that the
machinery ran, not that the output is worth having.

## The fix

Three changes, in `patches/0002-knowledge-writeback-string-evidence.patch`:

1. `_as_list()` — coerce `evidence` / `artifacts`; a bare string becomes a
   one-entry list instead of a character sequence.
2. Reject a candidate with a blank `statement` (`"no statement (TL;DR would be
   empty)"`) rather than writing a lesson that says nothing.
3. `_looks_like_path()` rejects tokens with no alphanumeric character.

Plus `scripts/tests/test_phase_state_knowledge_writeback.py` — 10 tests across
evidence normalization, statement validation, and the path heuristic.

Verified on a scratch clone:

| | before | after |
|---|---|---|
| new tests | 7 failed, 3 passed | **10 passed** |
| full suite | 791 passed | **801 passed** |
| `eval.sh` | 45/45 | **45/45** |

(The 1 unrelated failure in both columns is finding 02.)

## Recovering the damaged files

The corruption is mechanical, so most of it reverses — concatenate the
single-character bullets. `workshop/tools/recover-lessons.py` does this,
**read-only by default**:

```bash
python3 workshop/tools/recover-lessons.py --dir "$WRIT_DIR/.writ/knowledge"
```

All 10 recover substantive evidence. For example:

> `scripts/eval-leanness.py:527,533,540,603` pre-fix;
> `2026-08-11-governor-instrumentation` Story 1; verified post-fix by lowering a
> ceiling by 1 (warning returns naming the ceiling) and by injecting a legacy
> unbounded string (warns per-metric, silences nothing).

That is a real engineering note that currently reads as 257 bullets of loose
letters. `--write` rebuilds `related_artifacts` and `## Related` from the
recovered prose; verified 0 residual corruption across all 16 files, frontmatter
still parses, and the 6 hand-written lessons untouched.

**The `## TL;DR` is not recoverable** — it was never written. All 10 need a human
sentence. The titles survived and read like statements, which is a reasonable
starting point, but that's his call, not a script's.

Good news: the *evidence base* is intact. `load-report.md` survives in
`.writ/specs/archive/2026-08-12-disclosure-implement-story/`, and the 1,017-byte
figure is corroborated in five other artifacts. Only the lessons were damaged.

## Workshop questions

1. **Where does the candidate JSON come from?** The bug only fires when a
   producer emits `evidence` as a string. Since all 10 are damaged, the producer
   does it every time — so the schema contract between the phase-close step and
   `knowledge-writeback` is either undocumented or unenforced. Fixing the
   consumer is the right defensive move; the producer is the real bug.
2. **Should the writeback validate its own output?** It writes markdown with a
   known shape and never reads it back. A one-line assertion — no list item is a
   single character, TL;DR is non-empty — would have caught this on day one.
3. **What else has no unit tests?** `knowledge_writeback` was found by looking.
   `scripts/tests/` covers 792 cases but nobody has mapped what it *doesn't*
   cover. That map is probably worth more than the next ten tests.
