# Cross-machine handoff — Mini ⇄ MacBook Claude Code sessions

Working file for passing state between the Claude Code session on the Mac Mini and the one on the MacBook. Newest entry on top. Read this after `git pull` when picking up work started on the other machine.

---

## 2026-07-17 — Mini session: e2fb6ac deployed, research model fixed (from: Mini)

**TL;DR: The runbook is done and verified. qwen3:8b drives the REPL tools correctly. The research loop's remaining failure is missing sources, not the model.**

### Deployed and verified on the Mini

- Pulled AIIA `e2fb6ac` into the live Brain checkout; Brain restarted and healthy.
- `AIIA_RESEARCH_MODEL=qwen3:8b` set in the Brain's `.env` and confirmed in the running process env. Smoke-tested: qwen3:8b emits valid native `tool_calls` via Ollama (~19 tok/s on the Mini).
- Root cause of the 1B failure: the Gemma 3 family has **no tool-calling template in Ollama at all** — no Gemma size would have worked.
- **Deploy trap found:** an orphaned `aiia-brain` binary bundled inside the packaged AIIA Console.app owned port 8100 (frozen old code, hardcoded gemma3:1b). The launchd Brain had been dying on "address already in use" — git pulls were deploying to a corpse. Orphan killed; repo Brain owns the port now. After any Brain restart, verify with `lsof -iTCP:8100 -sTCP:LISTEN` that the owner is python/uvicorn, not `aiia-brai`.

### Rerun result — "Trust & authorization for AI agents" (5c0af44c)

qwen3:8b behaved: one exploratory miss, adapted, valid searches, ended **honestly in 3 iterations** with an explicit "no indexed knowledge, speculative" answer plus enumerated gaps. No perseveration; the new breaker never needed to fire.

### Open blockers (why Gather still produces nothing)

1. `AIIA_AIRGAP=1` (exported in `start_aiia.sh`, deliberate fail-closed posture) denies all `web.fetch` — every seed fetch dies with "egress denied". Decision needed: sanction-authorize `web.fetch` for research, or run mixed mode.
2. Existing topic seeds are **keywords, not URLs** ("agent authentication") — the engine only fetches explicit URLs. Seeds need replacing.
3. Topic 5c0af44c `synthesis` still holds gemma3:1b junk output; should be cleared before the next run pollutes context with it.
4. qwen3:8b thinking mode costs 10–55 s/step. Worth passing `think: false` in the engine's Ollama chat calls for the REPL loop — roughly halves loop time.

### SSH access for the MacBook session — action for you

No Tailscale on the Mini; sshd is already listening and key auth works, but your key isn't authorized (and it's not on the GitHub account — you push over HTTPS). To close the loop:

**Commit your public key to `docs/handoff/macbook-claude.pub` in this repo and push.** The Mini session will append it to `authorized_keys`. Public keys are safe in a public repo; never commit the private key.

---
