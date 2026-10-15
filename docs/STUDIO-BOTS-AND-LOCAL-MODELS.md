# Bots entry point and local model evaluation

## Scope (2026-10-14)

Studio opens at `#/bots`. Named bot cards show purpose, status and latest output;
selecting one opens its task form. Work and Projects remain primary destinations.
Today, Inbox, Jobs, Overview and configuration remain under More with their
existing deep links. Opening a bot never starts a run.

This slice uses the existing agent API and records. It does not create a new agent
framework or change assignments, tenancy, schedules, model defaults or egress.
It is based on `main` at `e5a3ab9`, independently of the delivery dashboard PR
#111 and morning-note PR #108. Reconcile their home/navigation changes before
merging both; this slice intentionally makes Bots the default destination.

## History and use cases follow-up (2026-10-15)

Cards show the last run time and a single-line outcome. Bot details show History,
newest first, including manual, interval, and assignment runs. Expand a row for
its task input and result or failure evidence. Older/Newer controls page through
the durable ledger, beyond the 12-run recent cache (20 runs per page). Missing
history is shown as unavailable with retry, separately from an empty history.

Each bot exposes `use_cases: [{"label": "...", "task": "..."}]` through the agent
configuration API. POST, PUT and PATCH accept up to three shortcuts (80-character
labels, 6,000-character task prompts). Omitting the field preserves existing
configuration on edits; an explicit empty list hides shortcuts. Existing records
without the field get three defaults at read time, without rewriting their stored
config. Named defaults reflect the current roster's descriptions; other bots use
their mission as the basis for brief, risk, and next-step tasks. New bots store
these defaults when created. Custom shortcuts override defaults.

Click a use-case button to run its task immediately. Optional **Additional
context** is appended as `\n\nAdditional context:\n<context>`. **Send task** submits
the text alone as a free-form task. The existing 8,000-character request limit,
busy/retired guards, draft preservation, model selection, and execution path apply.
A use case whose combined task would exceed the request limit is disabled.

Verification uses temporary SQLite/config files for backend contracts and mocked
API responses for desktop/mobile browser behavior. No live inference is needed.
Screenshots are in [the follow-up evidence](screenshots/bot-history-use-cases/).

## What the performance evidence says

The reported screenshot shows `qwen3:8b`, with 23.1 seconds of Ollama HTTP time.
That number excludes Studio context preparation and does not identify the
bottleneck. Studio already sends `think: false`; the local client currently uses
32,768 context tokens by default and keeps models resident for 30 minutes.

Repository/GitHub context preparation used synchronous subprocess calls from
the async request handler. It now runs in a worker thread, preserving the existing
execution lock. This keeps the event loop responsive; it does not reduce the
model's generation time. `/v1/chat` now exposes optional `timing` fields in
milliseconds: `total_ms`, `load_ms`, `prompt_eval_ms`, `eval_ms`. Missing or malformed
provider durations are unknown, not zero. These fields are not yet persisted in
the Studio ledger or displayed as an end-to-end latency breakdown.

Read-only inspection in this session reached the **MacBook**, not the Mini:

| Installed model | Quantization | Role in a future comparison |
|---|---|---|
| `qwen3:8b` | Q4_K_M | Current baseline |
| `gemma3:1b` | Q4_K_M | Small model for bounded extraction and short replies |
| `gemma3n:e4b` | Q4_K_M | Installed, but its package is larger than Qwen's |
| `llama3:8b` | Q4_0 | Another 8B baseline, lower priority |
| `nomic-embed-text:latest` | F16 | Embeddings, not a chat replacement |

No model was loaded at inspection. Mini hardware, inventory and performance remain
unverified. No model download, inference benchmark or configuration change was
performed for this slice.

## Proposed division of work

1. **Code supplies facts.** CI state, PR state, deployment evidence, counts and
   exact status lookups should not require a language model to rediscover them.
2. **Jev supplies optional semantic judgments.** For an ambiguous request, suggest
   a suitable bot or classify the kind of work. A person opening a specific bot
   already selected the handler and needs no routing call.
3. **A local LLM writes or reasons.** Give it relevant evidence and a bounded output
   budget for summaries, explanations and open-ended tasks.

This is a recommendation, not a newly enabled automatic routing architecture.
The existing TypeSafe integration is an explicit, consented cloud advisor with
egress authorization and confirmation before execution. Preserve those controls;
airgap stays on. Jev returns typed decisions and probabilities rather than prose.
Adding it before every local request would add another network call without
necessarily saving work. Evaluate routing accuracy and abstentions on labeled
examples before using its confidence to automate any decision.

References: [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one),
[intent routing](https://docs.typesafe.ai/patterns/intent-routing).

## Mini measurement procedure

Run from the checkout on the actual inference host, or use a loopback SSH tunnel.
The script rejects remote HTTP endpoints, never downloads models, and inventories
only unless `--run` is specified:

```sh
python3 scripts/benchmark_studio_models.py
python3 scripts/benchmark_studio_models.py --run --model qwen3:8b --num-ctx 4096
```

Use exact installed names from inventory. The opt-in command runs the same
synthetic three-bullet incident summary twice, reporting residency before each
request, wall time, provider durations, token counts, decode rate and output for
inspection. It is a smoke comparison, not a representative quality evaluation.
It does not force unload or alter persistent model settings. A first request is
not automatically a cold run; check residency and load duration.

For a decision, compare the same representative tasks across 4K and 8K contexts
(and the current 32K baseline when needed), equal output budgets and three warm
repetitions. Record hardware, Ollama version, model digest, quantization, loaded
context and available memory data. Check factuality, instruction following,
completion and failures before ranking median wall time. The non-streaming script
does not measure first-visible-token latency. Never shorten necessary input merely
to improve the score.

Keep `qwen3:8b` until there is a measured winner. Test installed `gemma3:1b` for
narrow tasks first. A later authorized download of `qwen3.5:4b` is the first newer,
smaller challenger worth evaluating; its Q4 package is about 3.3 GB versus Qwen3
8B's 5.2 GB. Package size is not runtime memory or a speed prediction. Use supported
API thinking controls, not prompt suffixes: Qwen3.5 does not support `/nothink`.

References: [Qwen3 8B](https://ollama.com/library/qwen3:8b),
[Qwen3.5 4B](https://ollama.com/library/qwen3.5:4b),
[Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[Gemma3 1B](https://ollama.com/library/gemma3:1b),
[Ollama response timings](https://docs.ollama.com/api/chat),
[context allocation](https://docs.ollama.com/context-length).
