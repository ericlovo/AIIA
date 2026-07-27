# Vendored: `local-llm`

`vendor/` is a byte-identical copy of an upstream project. **Do not edit anything under
`vendor/`.** AIIA's additions live beside it, never inside it.

| | |
|---|---|
| Upstream | https://github.com/jhammant/local-llm-skill |
| Pinned commit | `9a8f3621f9665e1ae4fabefd81a62d11cca605bf` |
| Commit date | 2026-07-27 |
| Commit subject | `feat(aiod): burst to a rented GPU — explicit consent required, every time` |
| Upstream version | 1.0.0 |
| License | MIT (see `vendor/LICENSE`) |
| Runtime | Node 18+, **zero runtime dependencies** |

## Why vendored rather than depended on

The tool is a self-contained ES-module CLI with no runtime deps and no npm publish. Vendoring
keeps the mini working offline and under air-gap, pins the exact behaviour AIIA's SKILL.md
describes, and makes an upstream change a reviewable diff instead of a silent version bump.

## Layout

```
.claude/skills/local-llm/
├── SKILL.md                     # ours — the AIIA-tuned skill Claude actually reads
├── UPSTREAM.md                  # ours — this file
├── bin/local-llm                # ours — shim, runs the vendored CLI without a global install
├── examples/endpoints.aiia.json # ours — endpoint registry defaulting to Ollama
└── vendor/                      # upstream @ 9a8f362, byte-identical
    ├── src/  test/  package.json  README.md  LICENSE  SPEC*.md  BENCHMARKS.md
    └── skill/SKILL.md           # upstream's own skill doc — superseded by ../SKILL.md
```

`vendor/skill/SKILL.md` is retained only to keep the copy byte-identical for diffing. The
skill Claude loads is `.claude/skills/local-llm/SKILL.md`, one level up.

## AIIA deviations from upstream's skill doc

Our `SKILL.md` is a rewrite, not a copy. Substantive differences, each deliberate:

1. **`local-llm agent` removed.** Upstream's `skill/SKILL.md` §3 documents an agentic-coding
   command (`local-llm agent -C <dir> --class coder …`). **No `agent` verb exists in the
   shipped CLI** — it is absent from `src/cli.mjs --help` and from the whole of `src/`.
   Carrying that section over would have told Claude to run a command that does not exist.
   Re-check this on every sync; if upstream implements it, restore the section.
2. **Ollama-first.** Upstream is LM Studio-first and its auto-probe lets LM Studio win the
   default when both backends answer. AIIA is an Ollama shop (`LOCAL_LLM_URL`, default
   `http://localhost:11434`), so ours documents pinning the default to Ollama and ships
   `examples/endpoints.aiia.json`.
3. **Brain boundary added.** A table splitting `local-llm` (batch, memory-bound, no context)
   from the Brain at `:8100` (`/v1/route`, `/v1/chat`, vault memory), plus two explicit
   consequences: local-llm runs bypass AIIA's purpose-attributed token metering
   (`/v1/tokens/today`) and never write vault memory.
4. **Air-gap interaction added.** `AIIA_AIRGAP=1` and `local_brain/egress.py` fail-closed
   denial govern the *Brain's* egress only. `local-llm` is a separate process, so burst is a
   cloud egress path AIIA's guard cannot see or block. Our doc makes burst-under-air-gap a
   stop-and-ask.
5. **Shared memory pool warning added.** The Brain's own resident Ollama models compete with
   batch jobs for the same RAM; a careless batch can evict a model the Brain is mid-flight on.
   Ours says to pin first.
6. **`.gitignore` interaction noted.** AIIA's root `.gitignore` covers `*.jsonl` and
   `reports/`, so batch artifacts never appear in `git status`. Documented so it reads as
   intent rather than a bug.
7. **Invocation via the shim** (`bin/local-llm`) rather than assuming a global `npm link`.

## Syncing a newer upstream

```bash
git clone https://github.com/jhammant/local-llm-skill /tmp/lls
diff -r --exclude=.git /tmp/lls .claude/skills/local-llm/vendor    # review every hunk
rsync -a --delete --exclude=.git /tmp/lls/ .claude/skills/local-llm/vendor/
(cd .claude/skills/local-llm/vendor && node --test)                # 102 tests, zero deps
```

Then update the pinned commit in the table above, re-read the deviation list — especially
item 1 — and re-run the sanitization guard (`.github/workflows/ci.yml`), since the guard
scans vendored files like any other.

## Verification at time of vendoring

- `node --test` in `vendor/`: **102 passed, 0 failed** (Node 22 on Linux; no network,
  no live backend — tests use injected clients and temp state).
- Sanitization guard banned-string scan over the vendored tree: **clean**.
- No absolute `/Users/…` or `/home/…` paths in code (two prose mentions in
  `vendor/SPEC-oss.md` only).
