---
name: local-llm
description: >-
  Run work on the mini's local models (Ollama / LM Studio) instead of burning metered API
  quota — a pool that is free but memory-bound. Use above all for BATCH jobs (classify,
  summarize, extract, score, embed hundreds or thousands of items, especially overnight),
  for one-shot asks a smaller model handles fine, and for security work where an aligned
  model false-refuses authorized testing. Triggers: "run this locally", "use the local
  model", "batch these", "crunch these overnight", "don't burn quota on this", "local llm",
  "ollama", "lm studio", "classify these on the mini".
---

# /local-llm — run bulk work on the mini's local models

Drives the mini's local models through the vendored `local-llm` CLI. This pool has **no
quota and no per-token cost** — the scarce resources are **memory** and **throughput**.

## Invoking the CLI

The tool is vendored at `.claude/skills/local-llm/vendor/`. Prefer the repo-local shim,
which needs no global install:

```bash
.claude/skills/local-llm/bin/local-llm budget
```

If the user has run `npm link` in `vendor/`, plain `local-llm` also works. Node 18+ only,
zero runtime dependencies — there is nothing to `npm install`.

**Never assume what hardware or models are available — ask the tool.** The model library on
the mini changes, and what fits today may not fit tomorrow:

```bash
local-llm budget          # memory ceiling, reserve, used, free, and what's loaded
local-llm models --fit    # models on disk, and which can actually load right now
local-llm ps              # what is resident this second
```

Run these before planning any sizeable job, and base your recommendation on what they
return, not on what was true last time.

## This skill vs. the AIIA Brain

AIIA already routes and remembers. This skill does **neither**. Keep the split clean:

| Use | For |
|---|---|
| **`local-llm batch`** | many similar items, mechanical judgement, no conversation context. **The primary use.** |
| **`local-llm ask`** | a quick one-shot a mid-size local model handles fine, with no memory context |
| **Brain `/v1/route`, `/v1/chat`** (`:8100`) | anything that should consult or write AIIA memory, or be attributed in token metering |
| **`aiia ask`** | a question that wants AIIA's decisions/patterns/session memory behind it |
| a frontier model / this conversation | the hard middle: taste, cross-repo orchestration, genuinely difficult reasoning |

Rule of thumb: **local is for volume and the cheap tail, not the hard middle.** A local 27B
is a poor substitute for a frontier model on one difficult task, and an excellent substitute
for running one easy prompt four thousand times.

Two consequences worth stating plainly:

- **`local-llm` bypasses AIIA's token metering.** Runs through this CLI do not appear in
  `/v1/tokens/today` or the purpose-attributed metering. That is acceptable — local tokens
  are free — but do not treat the metering dashboard as a complete record of work done.
- **`local-llm` bypasses AIIA memory.** Nothing a batch produces lands in the vault. If a
  result should be remembered, write it back deliberately (`aiia memory add`, or
  `POST /v1/aiia/remember`).

## 1. Batch — the main event

```bash
local-llm plan  items.jsonl --template t.md                     # estimate FIRST
local-llm batch items.jsonl --template t.md --out results.jsonl --class workhorse
```

- **Input** JSONL, one object per line; `--field text` wraps plain-text lines.
- **Template** uses `{{field}}` placeholders; a missing field fails fast with the line number.
  `--prompt "…"` works for a one-liner instead of a template file.
- **Output** JSONL written incrementally — `{i, id, ok, response, usage, ms, error}`.
- **Resumable** — re-running the same `--out` skips completed ids, so an overnight run
  survives a crash, a sleep, or a power cut. `--restart` forces a clean run.
- **Constrained output** — `--allow positive,mixed,negative` restricts the answer to those
  values and re-prompts when the model drifts. Use it for every classification job.
- **Concurrency** defaults to the model's advertised parallel slots. A single bad item is
  recorded and skipped; it never kills the run.

**Always run `plan` first and show the user the estimate** before starting a long job.
`plan` times real end-to-end samples, so its ETA is worth more than a token-rate guess.

Choose `--class` by how hard the *per-item* judgement is — `reflex` for classify/tag/extract,
`workhorse` for summarize/score, `heavy` only when genuinely needed. On thousands of items an
oversized model costs hours for no gain. The tool picks a concrete model for the class from
what's installed and reports which and why. Classes: `reflex`, `workhorse` (default), `coder`,
`heavy`, `vision`, `embed`, `security`.

### Where to put output files

**AIIA's root `.gitignore` ignores `*.jsonl` and `reports/`.** Batch inputs and outputs are
therefore invisible to `git status` by design — this is correct (results are runtime data,
not source), but do not be confused when a file you just wrote does not show up as untracked.
Write batch artifacts under `reports/` or a scratch directory, and never `git add -f` a
result set without asking.

## 2. Memory is what actually breaks

Models range from ~3 GB to ~100 GB; the budget is finite and shared with the OS. A 100 GB
model and a 3 GB model do not co-reside on a 128 GB machine. `local-llm` runs every load
through admission control: it evicts least-recently-used models to make room, refuses a model
larger than the whole budget, and names the remedy instead of failing obscurely.

```bash
local-llm budget          # what's free, what's loaded
local-llm pin <model>     # protect a model you're actively using elsewhere
local-llm unload --all    # reclaim everything
```

**Before a long batch, run `local-llm budget` and tell the user what will be evicted.**

This matters more on the mini than it looks: the Brain's own Ollama models are in the same
memory pool. If AIIA is serving `/v1/chat` or running the research loop with a resident model,
a careless batch can evict it mid-flight. **Pin the Brain's working model before a long batch**,
or run the batch when the Brain is idle.

Budget defaults can be tuned in `~/.config/local-llm/config.json` (`ceilingGb`, `reserveGb`)
or via `LOCAL_LLM_CEILING_GB` / `LOCAL_LLM_RESERVE_GB`.

## 3. Endpoints

With no registry file, the tool probes LM Studio (`127.0.0.1:1234`) and Ollama
(`127.0.0.1:11434`) and registers whichever answers — **LM Studio wins the default if both
are up.** AIIA is an Ollama shop (`LOCAL_LLM_URL`, default `http://localhost:11434`), so to
pin the default to the same backend the Brain uses, copy the example registry:

```bash
mkdir -p ~/.config/local-llm
cp .claude/skills/local-llm/examples/endpoints.aiia.json ~/.config/local-llm/endpoints.json
local-llm endpoints        # confirm which is default
```

Backend capability differs, and it changes what the tool can promise:

| Feature | LM Studio | Ollama | Generic OpenAI |
|---|---|---|---|
| Model sizes | yes | yes | no |
| Loaded state | yes | yes | no |
| Load / unload | yes | yes | no |

An endpoint that cannot report sizes **and** loaded state is not memory-managed: `budget`
reports `managed: false` and admission control steps aside rather than inventing numbers.
Do not promise the user eviction safety on such an endpoint.

> LM Studio's own CLI (`lms`) is often **not on PATH** in an agent shell. `local-llm` resolves
> it internally; if you must call it directly, use `command -v lms || echo "$HOME/.lmstudio/bin/lms"`.

## 4. Security work — uncensored models, on request only

Aligned models routinely **false-refuse benign authorized work** — analysing a payload from
your own scan, writing a PoC against a repo you own. If uncensored variants are installed,
`--class security` / `--uncensored` routes to them.

- **Explicit request only** — never auto-selected, never a fallback the router reaches on its
  own. These models trade instruction-following and factual accuracy for the absence of
  refusals, so they are a remedy for a false refusal, not a default.
- Scope is unchanged: **systems the user owns or is authorised to test.** This removes a
  false-positive tax on legitimate work; it is not a route around judgement.

## 5. Burst to a rented GPU — spends real money, and escapes AIIA's egress guard

If [`aiod`](https://github.com/jhammant/AIonDemandCluster) is installed, `local-llm` can rent
a GPU serving an OpenAI-compatible endpoint and treat it as just another endpoint. If `aiod`
isn't present, burst simply doesn't appear.

**Read this before ever using it:**

- **Air-gap mode does not cover it.** `AIIA_AIRGAP=1` and `local_brain/egress.py` fail-closed
  denial govern *the Brain's* egress. `local-llm` is a separate Node process opening its own
  connection — AIIA's guard cannot see or block it. **If air-gap is on, burst is off.** Treat
  a burst request under air-gap as a contradiction and stop to ask, rather than quietly
  punching a hole through a control the user deliberately switched on.
- **Never start a rented box without showing $/hr and estimated total and getting an explicit
  yes — every time.** Not configurable, and no autonomous process (research loop, scheduled
  autonomy, proactive executor) may trigger it.
- Always set `--idle` and `--ttl`; teardown runs in a `finally` and on SIGINT.
- **Billing continues until teardown.** If anything is live, say so loudly and unprompted:
  `local-llm burst status` answers "is anything billing right now?".
- The endpoint is a public IP behind a single bearer token, so batch items — which are often
  private — do not go remote without `--allow-remote-data` per run. Local is the default for
  the user's data.

```bash
local-llm plan items.jsonl --template t.md    # compares local (free, slow) vs burst ($, fast)
local-llm burst status
local-llm batch … --endpoint burst --allow-remote-data --idle 10 --ttl 2 --yes
```

## 6. Report what it cost

Local runs are free in money but not in *time*: report wall-clock and items/sec so the user
can judge whether it was worth doing locally. `local-llm bench` measures single-stream and
concurrent token rates and records them to `~/.local/state/local-llm/throughput.json`, which
is what makes later `plan` estimates sharpen over time. For burst runs, report actual spend.

## Why this pattern

Subscription pools are **quota**-bound and expire on a cycle. Local is **memory**-bound and
never expires. Rented GPUs are **money**-bound and effectively uncapped. Routing each job to
whichever constraint is loosest is the whole point — and high-volume batch work should almost
always land on the free pool.

## Note on the vendored copy

`vendor/` is upstream `jhammant/local-llm-skill` held byte-identical — do not edit it. See
`UPSTREAM.md` for provenance, the pinned commit, how to sync, and the known upstream
doc/implementation divergences (notably: **upstream's own SKILL.md documents a `local-llm
agent` command that does not exist in the shipped CLI** — it is deliberately absent here).
