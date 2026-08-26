# Finding 02 — `ac-trace.py` crashes on a symlink loop (Python ≤ 3.12)

**Severity:** medium — real crash, in the citation scanner `verify-spec` depends on
**Where:** `scripts/ac-trace.py:419-423` and `:434-438`
**Status:** confirmed. Fix written and verified — **deliberately not applied.**
The checkout stays pristine; the patch is a proposal for him, not a change we made.

## What happens

Writ ships a test for exactly this crash. It fails:

```bash
python3 -m pytest scripts/tests/ -q
# 1 failed, 791 passed, 3 skipped
# FAILED test_ac_trace.py::CitationScanTests::test_symlink_loop_does_not_crash_the_scan
```

```
RuntimeError: Symlink loop from '/tmp/tmpkg_a4g0_/loop'
  scripts/ac-trace.py:436  in _walk_candidates
      resolved = file_path.resolve()
```

## Root cause

The walker guards `resolve()` against the wrong exception type:

```python
if file_path.is_symlink():
    try:
        resolved = file_path.resolve()
    except OSError:          # <-- never fires for a symlink loop
        continue
```

On Python ≤ 3.12, `Path.resolve()` converts the underlying `ELOOP` into a
**`RuntimeError`**, and `RuntimeError` is not a subclass of `OSError` — so the
`except` clause doesn't catch it and the exception escapes. Python 3.13 removed
that conversion; `resolve()` returns the path without raising.

Confirmed directly:

```
python 3.11.15
Path.resolve()                   -> RuntimeError: Symlink loop from '/tmp/.../loop'
os.path.realpath(strict=False)   -> ok: /tmp/.../loop
RuntimeError is OSError subclass? False
```

So the behavior is **Python-version-dependent**: green on 3.13+, crashing on
3.10–3.12. Writ's `pyproject`-equivalent floor isn't pinned anywhere in the repo,
and the eval harness doesn't record which interpreter it ran under.

## Why nobody caught it

`scripts/tests/` is not wired into CI. Writ's own leanness ledger says so
plainly, in the justification for the `scripts` surface growth:

> "CI runs `scripts/eval.sh` and never `scripts/tests/`, so each checker's eval
> scenarios plus its `require_literal`/`forbid_literal` bindings are its entire
> CI protection"

That's 792 tests with no gate in front of them. This finding is what that gap
costs: a test written on purpose, for a crash that's real, sitting red.

The same ledger entry records the symmetric lesson already learned once —
"the per-command byte ratchet in `test_governor_enforcement.py` is NOT wired into
`eval.sh` and caught a real regression only because the full unit suite was run
by hand."

## The fix

Both symlink guards, dir and file:

```python
except (OSError, RuntimeError):
    # Python <=3.12 raises RuntimeError (not OSError) for a
    # symlink loop; 3.13+ resolves it without raising.
    continue
```

Verified on Python 3.11.15 against the real repo:

| | before | after |
|---|---|---|
| `pytest scripts/tests/test_ac_trace.py` | 1 failed, 48 passed | **50 passed** |
| `pytest scripts/tests/` | 1 failed, 791 passed | **792 passed** |
| `eval.sh --check=ac-trace` | PASS (20/20) | **PASS (20/20)** |

Patch: `workshop/patches/0001-ac-trace-symlink-loop-runtimeerror.patch`.
Apply and revert, both verified on a clean clone:

```bash
# absolute path — git -C resolves relative paths against the TARGET repo
git -C "$WRIT_DIR" apply "$PWD/workshop/patches/0001-ac-trace-symlink-loop-runtimeerror.patch"
bash workshop/run-harness.sh tests                   # 792 passed
git -C "$WRIT_DIR" checkout -- scripts/ac-trace.py   # pristine again
```

It is not applied by default. His tree starts and stays as shipped unless he
says otherwise.

`os.path.realpath(path)` is the alternative — it never raises for loops on any
version — but it changes the return type and skips `Path`'s normalization, so the
two-exception catch is the smaller change.

## Sibling sites

An AST sweep for `resolve()` inside a `try` whose handlers omit `RuntimeError`
found seven more. Only the two in `ac-trace.py` are symlink guards; the rest
catch `ValueError` from `relative_to()` and are unrelated. Listed for the room:

```
ac-trace.py:420, :435                    <- the bug (symlink guard)
eval-recommend-state-adversarial.py:396  <- bare except, already broad
recommend-state.py:464, :482, :1345, :1494
revert-resolve.py:461
test-integrity.py:270
```

## Workshop questions

1. **Wire `scripts/tests/` into CI, or not?** The leanness ledger treats the gap
   as a deliberate cost of the eval-scenario model. But this bug is the second
   documented escape through it. If the answer is "not," what makes the eval
   scenarios catch a version-dependent crash they currently can't see?
2. **Pin the interpreter floor.** The harness reports bytes, lines, findings, and
   scenarios — but never the Python version it ran under. A version-dependent
   result that isn't recorded is unreproducible by definition.
