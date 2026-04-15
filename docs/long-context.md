# Long-Context Support on AIIA

AIIA can handle context windows up to 64K tokens (and more — see ceiling
below) on a Mac Mini with 24GB unified memory. This doc explains how to
enable long context, the trade-offs involved, and the measured
behavior on real hardware.

---

## The short version

Set these three Ollama environment variables before starting the
Ollama service, and AIIA's routing model gets 64K context with
reasonable load times:

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
export OLLAMA_KEEP_ALIVE=24h
```

Then pass `num_ctx: 65536` on the request (or set `AIIA_NUM_CTX=65536`
for the Local Brain API to thread it through automatically).

That's it. No sidecar process, no custom llama.cpp build, no MLX
backend, no model-swap juggling. Ollama does it natively once you
turn on flash attention and quantized KV cache.

---

## Why you might want this

AIIA's default routing model (llama3.1 or gemma4 families) comfortably
handles 4K–8K context for the common case: a chat message plus some
retrieved knowledge chunks, run through SmartConductor to pick a route
and domain.

Long context is for the less-common case where a single request needs
the model to hold a lot in its head at once:

| Use case | Why you need > 8K |
|---|---|
| **Story decomposition** with the full codebase as context | Letting AIIA see every relevant file when it decomposes a P1 story into action items. Fewer hallucinated function names, fewer wrong file paths. |
| **Morning briefing** synthesis across a week of standup notes + commits | A week of development activity can easily fill 32K tokens of markdown + code. Summarizing it in one pass beats summarizing in chunks that lose cross-reference. |
| **Multi-document RAG** for research-flavored queries | "Compare how modules A and B handle authentication" works better with both files fully loaded than with chunked retrieval. |
| **Agent self-review** before queuing an action for autonomous execution | Loading the diff plus the files it touches plus the story it's responding to — easily 16K+ on a real refactor. |
| **A2A planner agents** calling several mesh agents and synthesizing their results | Inbound task + aiia-ask response + aiia-search results + log-story context add up fast. |

If your AIIA workload is mostly routing short chats, you don't need
this. If your AIIA workload includes any of the above, you probably do.

---

## The Ollama env vars, explained

### `OLLAMA_FLASH_ATTENTION=1`

Required for quantized KV cache to work at all. Flash attention is the
memory-efficient attention kernel; without it, `OLLAMA_KV_CACHE_TYPE`
is silently ignored and the cache uses default FP16 regardless.

No quality impact. Enables the next variable.

### `OLLAMA_KV_CACHE_TYPE=q4_0`

Compresses the key/value cache to 4-bit integers instead of 16-bit
floats. Approximately 4x memory reduction on the KV cache. At 64K
context on a Gemma 4 E4B-class model, this drops the cache footprint
from roughly 8GB to roughly 2GB of wired memory.

Trade-off: about 0.5% increase in perplexity on standard language
modeling benchmarks. For routing, summarization, and code assistance
workloads, this is well below the noise floor. For high-precision tasks
like formal proofs or exact math, measure for yourself.

Alternative values:
- `q8_0` — 8-bit, approximately 2x memory reduction, near-zero quality
  delta. Use this if you need the safest quality posture.
- `q4_0` — the one we recommend. Best memory/quality balance.

### `OLLAMA_KEEP_ALIVE=24h`

Keeps the model resident in memory between requests. Default is
5 minutes. At 64K context, cold-loading the model plus allocating the
KV cache buffer takes measurable time (see benchmarks below). You do
NOT want to pay that cost on every request. `24h` effectively makes the
model permanent until the OS reclaims its memory under pressure.

---

## Measured performance

Benchmark hardware:
- Mac Mini M4, 24GB unified memory
- macOS Sequoia
- Ollama HEAD-96b202d (pre-release build, April 2026)
- Gemma 4 E4B-Q4_K_M (approximately 5.4GB weights)

### Cold load time at 64K context

| KV cache | Wall time | Wired memory added |
|---|---|---|
| FP16 (default) | 685 seconds (11.5 minutes) | ~8 GB |
| `q4_0` + flash attention | 8.5 seconds | ~2 GB |

80x faster cold load with `q4_0`. The huge FP16 number isn't purely
due to KV allocation — it includes memory-pressure thrashing as the
system swaps other processes to make room. On a fresh, unpressured
system the FP16 cold load is probably 30–60 seconds. Even then,
`q4_0` at 8.5 seconds is obviously the right choice.

### Generation speed during long context

| Context filled | Tokens per second | Notes |
|---|---|---|
| 4K | ~28 tok/s | Same as default |
| 16K | ~27 tok/s | Near-linear, barely affected |
| 32K | ~26 tok/s | Still comfortable |
| 64K | ~24 tok/s | Prompt processing slower, output speed roughly constant |

KV cache quantization does not affect output speed meaningfully — the
model still runs at native weights. What changes is the memory
footprint and the time to populate the cache on the first prompt.

### Steady-state memory budget at 64K context

| Component | Memory |
|---|---|
| macOS + system processes | ~3.5 GB |
| Ollama runner + weights (Gemma 4 E4B Q4_K_M) | ~5.4 GB |
| KV cache at 64K tokens with `q4_0` | ~2.0 GB |
| nomic-embed-text (ChromaDB RAG embeddings) | ~0.5 GB |
| Brain API + Command Center Python processes | ~0.3 GB |
| ChromaDB in-memory index | ~0.5 GB |
| **Total resident** | **~12.2 GB** |
| **Free headroom on 24 GB Mac** | **~11.8 GB** |

That headroom matters. It means a 14B deep reasoning model
(`deepseek-r1:14b`) can swap in for nightly morning briefings without
evicting Gemma 4. See the ADR on Inference Routing for the model swap
strategy.

---

## Enabling on macOS with launchd

If you run Ollama as a `brew services` or `launchd`-managed daemon on
macOS, environment variables in your shell profile (`~/.zshrc`,
`~/.bashrc`) **do not** reach the daemon. You need to set them at the
`launchctl` level and then restart Ollama.

```bash
# Set the env vars at the launchctl user-domain level
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0
launchctl setenv OLLAMA_KEEP_ALIVE 24h

# Restart Ollama to pick up the new env
brew services restart ollama    # if installed via Homebrew
# or, if you launch it manually:
launchctl unload ~/Library/LaunchAgents/ollama.plist
launchctl load ~/Library/LaunchAgents/ollama.plist
```

**Caveat: `launchctl setenv` is session-level.** Its values survive
until the next reboot. After a reboot, you need to re-set them (or
bake them into a custom LaunchAgent plist — see below).

### Persistent LaunchAgent plist

Drop this at `~/Library/LaunchAgents/com.aiia.ollama-env.plist` and
`launchctl load` it once. It sets the env vars at every login so
Ollama (started after it) sees them.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aiia.ollama-env</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>launchctl setenv OLLAMA_FLASH_ATTENTION 1 &amp;&amp; launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0 &amp;&amp; launchctl setenv OLLAMA_KEEP_ALIVE 24h</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
```

Then:

```bash
launchctl load ~/Library/LaunchAgents/com.aiia.ollama-env.plist
brew services restart ollama
```

This plist is a one-shot setter: it runs `launchctl setenv` at login,
then exits. `KeepAlive` is false because we don't want it to stay
resident. The env vars persist for the rest of the session.

---

## Enabling with Docker Compose

If you run AIIA and Ollama in containers (see `docker-compose.yml` at
the repo root), set the env vars on the `ollama` service:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    environment:
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_KV_CACHE_TYPE=q4_0
      - OLLAMA_KEEP_ALIVE=24h
```

Then `docker compose restart ollama`. The AIIA service on port 8100
doesn't need any changes — it just sends requests with a bigger
`num_ctx` and the Ollama backend handles the rest.

---

## How AIIA uses it

AIIA's `LocalBrainConfig.models["routing"]` has a `max_tokens` field
for per-request limits but doesn't set `num_ctx` directly. To thread
long context through to Ollama for specific request types, pass the
context size in the `options` dict of the Ollama API call:

```python
response = await ollama_client.chat(
    model="gemma4:e4b",
    messages=[...],
    options={"num_ctx": 65536},
)
```

The Local Brain API's `/v1/aiia/ask` and `/v1/aiia/ask/stream`
endpoints accept an optional `num_ctx` field in the request body and
forward it to Ollama. You don't need to reconfigure AIIA — you just
pass the bigger `num_ctx` on requests that need it.

For the common short-chat path, leave `num_ctx` unset and Ollama uses
its default (typically 8K). Both paths share the same warm model
instance as long as `OLLAMA_KEEP_ALIVE` keeps it resident.

---

## The ceiling: 128K context

Gemma 4 E4B-class models technically support up to 128K tokens of
context. At 128K on `q4_0`, the KV cache is approximately 4 GB —
tight but fits on a 24GB Mac once you account for everything else.

We haven't benchmarked 128K routinely because:

- Filling a 128K context takes longer and the memory pressure is real.
- Most AIIA workloads hit the 64K ceiling naturally (a week of activity
  or an entire repo) and 128K is rarely needed.
- Cold-loading the 128K KV cache buffer, even with `q4_0`, takes
  around 30 seconds empirically (untested, extrapolated from 8.5s at
  64K).

If you need 128K regularly, consider:

- Moving Gemma 4 E4B to a dedicated TurboQuant sidecar that keeps the
  cache warm and uses the more aggressive `turbo2` 2-bit KV compression
  (roughly 6x vs `q4_0`'s 4x). See the sidecar pattern in the
  [Long-Context Scaling ADR](https://github.com/ericlovo/AIIA/issues).
  This is out of scope for AIIA's default runtime but is supported as
  an optional replacement for the Ollama backend.
- Or use Ollama's `num_ctx=131072` at your own risk and watch the
  first-load latency carefully.

---

## Troubleshooting

**Symptom: requests with large `num_ctx` take minutes to return the
first token.**

Likely cause: model is cold-loading with a fresh KV cache allocation.
Confirm by running the same request twice — the second request should
be fast if the model stays warm. If both are slow, check that
`OLLAMA_KEEP_ALIVE=24h` is actually reaching Ollama (`curl
http://localhost:11434/api/ps` — the model should have a far-future
`expires_at`).

**Symptom: `OLLAMA_KV_CACHE_TYPE=q4_0` has no effect, memory usage
unchanged.**

Likely cause: `OLLAMA_FLASH_ATTENTION` is not also set. Quantized KV
cache requires flash attention to be on. Ollama silently falls back
to FP16 if flash attention isn't enabled.

**Symptom: memory pressure, swap thrashing, slow Mac.**

Likely cause: the first long-context request is cold-allocating the
KV cache buffer while the system is under memory pressure from other
processes. Close browsers and idle apps, run the first long-context
request on a quiet system, and let the model stay warm. Subsequent
requests will be fast.

**Symptom: perplexity regression on a downstream evaluation you care
about.**

Likely cause: `q4_0` is too aggressive for your workload. Try `q8_0`
instead — it gives you half the memory savings but essentially zero
quality delta. Measure both on your actual task, not synthetic
benchmarks.

---

## Further reading

- [Ollama KV cache quantization PR](https://github.com/ollama/ollama/pull/8541) — upstream implementation
- [Flash Attention paper](https://arxiv.org/abs/2205.14135) — the kernel
- [TurboQuant KV compression paper](https://arxiv.org/abs/2504.14007) — the more aggressive sidecar option
- [AIIA Inference Routing ADR](../README.md#inference-routing) — how AIIA picks which backend handles a request

---

*Benchmarks in this document are empirical measurements from a specific
Mac Mini M4 with 24GB of unified memory running Ollama HEAD-96b202d in
April 2026. Your numbers will vary with model choice, quantization,
hardware, and system load. Measure your own workload.*
