# Security Architecture

Where AIIA's security mechanisms live, how they relate, and what each one does. Use this when you're trying to figure out "is X covered?" or "where would I add Y?"

For *reporting* a vulnerability, see [SECURITY.md](../SECURITY.md). For *running* the local scan suite, see [docs/security.md](./security.md).

## Layers

```
                                ┌─────────────────────────────────┐
                                │  Reporting & policy             │
                                │  - SECURITY.md (this repo)      │
                                │  - SECURITY.md (aiia-console)   │
                                └─────────────────────────────────┘
                                                ▲
                                                │
   ┌────────────────────────────────────────────┼─────────────────────────────┐
   │ Runtime defenses                                                         │
   │                                                                          │
   │   AIIA Brain (this repo)         AIIA Console (aiia-console)             │
   │   ──────────────────────         ───────────────────────────             │
   │   local_brain/execution/         src-tauri/src/keystore.rs               │
   │   - safety.py (pre-gate)         - URL allowlist + scheme guard          │
   │   - executor.py (3-tier)         - Per-provider auth, never to JS layer  │
   │   - strategies.py (AUTO /        - 0600 file perms, 0700 dir             │
   │     SUPERVISED / GATED)                                                  │
   │   - verification.py (post)                                               │
   │   - git_ops.py (safe git)                                                │
   │                                                                          │
   └──────────────────────────────────────────────────────────────────────────┘
                                                ▲
                                                │
   ┌────────────────────────────────────────────┼─────────────────────────────┐
   │ Build-time scans                                                         │
   │                                                                          │
   │   Local (`scripts/security_scan.sh`)       CI (`.github/workflows/`)     │
   │   ─────────────────────────────────        ─────────────────────────     │
   │   7 scanners: trufflehog, trivy,           ci.yml — ruff, sanitization   │
   │   bandit, semgrep, shellcheck,             security.yml — bandit, pip-   │
   │   hadolint, pip-audit. Baseline             audit (aiia); npm + cargo    │
   │   filtered via .security-baseline.json.    audit (aiia-console)          │
   │   See docs/security.md.                    gitleaks.yml — daily secret   │
   │                                            scan.                         │
   │                                                                          │
   └──────────────────────────────────────────────────────────────────────────┘
```

## Runtime defenses

### Brain — 3-tier execution model

Anything the Brain does that touches the filesystem, network, or shell goes through `local_brain/execution/` first. The pipeline is:

1. **`safety.py`** — pre-execution gate. Pattern-matches the proposed action against forbidden categories (deletion of unbacked dirs, shell injection patterns, etc.) and decides which tier the action requires.
2. **`strategies.py` + `executor.py`** — runs the action under one of three tiers:
   - **AUTO** — low-risk, runs silently (e.g. file reads inside the workspace)
   - **SUPERVISED** — runs but emits an event the user can intervene on
   - **GATED** — blocks on explicit human approval before running
3. **`verification.py`** — post-execution check. Confirms the action matched its declared intent (e.g. a "write to docs/" action didn't touch `local_brain/`).
4. **`git_ops.py`** — git-aware safe operations. Refuses to overwrite uncommitted changes, prefers branches over force-pushes, etc.

The tier model is defense-in-depth: even if a model proposes a destructive action, the gate refuses it; even if the gate misclassifies, the verification step catches drift. It's not perfect — a determined adversarial prompt could probably find paths — but it raises the floor significantly.

### Console — keystore + URL allowlist

The desktop app holds provider API keys (Anthropic, OpenAI, Google, DeepSeek, Moonshot) in `src-tauri/src/keystore.rs`. Properties:

- **JS can't read plaintext keys back.** It can only check presence (`keystore_get_keys`), set/clear (`keystore_set_key`, `keystore_delete_key`), or invoke a streaming call (`keystore_call`).
- **URL allowlist on every signed call.** `keystore_call` rejects any URL whose host isn't in the provider's allowlist or whose scheme isn't `https`. See `check_url_allowed` in `keystore.rs`.
- **0600 file / 0700 dir.** Keys at rest in `~/.aiia/keys.json` are readable only by the owning user (on Unix).
- **Per-call request_id.** Streaming events are scoped per-call so JS can't conflate responses.

### Known runtime gaps (tracked)

| Area | Gap | Mitigation today | Right answer |
|---|---|---|---|
| Console | Plaintext keys at rest | FS perms 0600; user-disk encryption (FileVault, LUKS) | OS keychain via `keyring` crate |
| Console | `String` holding key isn't `zeroize`d | None | `zeroize::Zeroizing<String>` |
| Console | Error body up to 400 chars in failure message | Truncated, not redacted | Strip provider-specific token patterns |
| Brain | Tier model can be coerced by adversarial prompts | Multiple layers (gate + tier + verify) | Out-of-band confirmation for GATED tier |

These are all bounded — none of them is "the lock is broken." But they're known and named here so the next contributor doesn't have to rediscover them.

## Build-time scans

### Local — `scripts/security_scan.sh`

The 7-scanner suite. Real, working, documented in `docs/security.md`. Quick recap:

| Scanner | What it catches |
|---|---|
| trufflehog | committed secrets, high-entropy strings |
| trivy | dep CVEs (multi-ecosystem) |
| bandit | Python SAST (e.g. `pickle.loads`, `subprocess` w/ `shell=True`) |
| semgrep | pattern-based static analysis (any language) |
| shellcheck | shell-script issues (the `scripts/` directory) |
| hadolint | Dockerfile linting |
| pip-audit | Python dep CVEs via OSV |

Baseline filter: `.security-baseline.json` lists accepted findings with reasons. The script exits non-zero on **new** findings only. See `docs/security.md` for the schema.

### CI — `.github/workflows/`

Three workflows enforce the security posture per-PR / per-push / on schedule:

| Workflow | Triggers | What it runs | Enforced? |
|---|---|---|---|
| `ci.yml` | push, PR | ruff lint + format; sanitization guard | yes (PR #19) |
| `security.yml` | push, PR, weekly cron | bandit (PR #22), pip-audit | bandit yes; pip-audit advisory |
| `gitleaks.yml` | push, PR, daily cron | gitleaks-action v2 | yes (PR #23) |

The CI is **complementary** to the local suite, not a replacement:

- **Local suite** is broader (7 scanners) and supports baseline suppression. Run it before opening a PR.
- **CI** runs a focused subset on every PR so nothing slips through review.

If a finding shows up in CI that the local suite would baseline-out, add it to `.security-baseline.json` *and* the relevant CI exclusion (`pyproject.toml` for bandit, `--skip` for runtime args). The two should track each other.

## Reading order for a new contributor

1. [SECURITY.md](../SECURITY.md) — how to report
2. This doc — what's in place
3. [docs/security.md](./security.md) — how to run scans locally
4. `local_brain/execution/` — read `safety.py`, then `strategies.py`, then `executor.py`
5. `aiia-console/src-tauri/src/keystore.rs` — read `check_url_allowed` and `keystore_call`

## When to update this doc

- A new runtime defense layer lands → add a row to the table
- A known gap gets closed → strike it through (don't delete; future readers will look for it)
- A new CI workflow lands → add it to the build-time table
- A scanner is added/removed from the local suite → update `docs/security.md` and reference here

Don't let this doc drift. It's the one place where someone landing on the repo can understand the whole security story without grepping.
