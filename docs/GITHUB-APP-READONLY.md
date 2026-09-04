# Read-only GitHub App for Agent Studio

GitHub access for AIIA **agents** is a separate path from the human
operator's `gh` CLI. Agents may read GitHub only through a dedicated
**GitHub App** installed on the repositories AIIA should see. They never
inherit a personal token, user OAuth session, or write scope.

Mounted local clones (**Repository read**) are unrelated: those are
filesystem mounts on the Mini. This document is the remote GitHub API
path (**GitHub read**).

## Why a GitHub App, not a PAT

| Credential | What it is | Why agents must not use it |
|---|---|---|
| Personal access token / `gh auth` user OAuth | The owner's identity, usually with write (and often org) scope | A prompt-injected agent could push, comment, or read private repos the owner can see. Revoking it also locks the human out of `gh`. |
| GitHub App installation token | Short-lived (~1h) token bound to **one App install**, with the permissions you granted, on **only the repos you selected** | Blast radius is the allowlist. Read-only permissions cannot write even if the model tries. Rotation is automatic. |

AIIA therefore:

- Reports Agent Studio GitHub status from **App credentials only**.
- Treats a logged-in `gh` CLI as a **diagnostic** (`cli_user_present`) — it does **not** flip status to `connected`.
- Refuses to mint or send a user token on the agent run path.

## Recommended App permissions

Create the App under the GitHub account or org that owns the AIIA repos.
Keep permissions **minimal**:

| Permission | Access | Why |
|---|---|---|
| Repository **contents** | Read-only | Tree, blobs, README, file contents |
| Repository **metadata** | Read-only (mandatory) | Repo list for the install |
| **Pull requests** | Read-only (optional) | PR titles/bodies for read tools |
| **Issues** | Read-only (optional) | Issue titles/bodies for read tools |

Do **not** grant: contents write, pull-request write, issues write,
workflows, administration, secrets, or any account/org write permission.

Subscribe to **no events** unless you later add a webhook consumer.
This App is a read client, not an automation bot.

## Install scope

Install the App **only** on the repositories AIIA already mounts or
explicitly allowlists. On the author's Mini that is typically:

- `ericlovo/AIIA` (this Brain)
- companion repos the Studio **Repository read** picker already exposes
  (console, Sanction, public proxy) — only if agents should see their
  GitHub metadata as well as the local clone

Do not install on every repository the owner can access. An App installed
on "All repositories" is a PAT with extra steps.

Optional allowlist (does not replace GitHub's install picker; it is
copied into Studio status + the agent prompt so models know the bound):

```bash
AIIA_GITHUB_APP_REPOS=ericlovo/AIIA,ericlovo/aiia-console
```

or in `~/.aiia/github-app.json`:

```json
{
  "app_id": "123456",
  "installation_id": "789012",
  "private_key_path": "~/.aiia/github-app.pem",
  "repositories": ["ericlovo/AIIA", "ericlovo/aiia-console"]
}
```

## Create and install (owner runbook)

1. GitHub → **Settings → Developer settings → GitHub Apps → New GitHub App**.
2. Name it something like `AIIA Agent Read` (must be globally unique).
3. Homepage URL can be the AIIA repo. Webhook: **inactive**.
4. Repository permissions: contents **Read-only**, metadata **Read-only**;
   optionally issues / pull requests **Read-only**. Everything else **No access**.
5. Where can this GitHub App be installed? **Only on this account**.
6. Create the App. Note the **App ID**.
7. **Generate a private key**. Download the `.pem`. Store it at
   `~/.aiia/github-app.pem` with mode `0600`. Never commit it.
8. **Install App** → select **Only select repositories** → pick the AIIA
   allowlist → Install.
9. From the install URL (`…/settings/installations/<id>`) copy the
   **installation ID** (the number in the URL).

## How Mini stores App credentials

Follow the same secret pattern as other Brain keys: environment (preferred
for launchd / systemd) or files under `~/.aiia/` with `0600`. Never commit
secrets; `*.pem` is gitignored.

### Environment (preferred)

```bash
# .env or launchd plist — not the repo
AIIA_GITHUB_APP_ID=123456
AIIA_GITHUB_APP_INSTALLATION_ID=789012
AIIA_GITHUB_APP_PRIVATE_KEY_PATH=~/.aiia/github-app.pem
# Optional: inline PEM with literal \n (discouraged; path is safer)
# AIIA_GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."
```

`AIIA_GITHUB_APP_CONFIG` may point at an alternate JSON path.

### Files

Default locations:

| Path | Mode | Contents |
|---|---|---|
| `~/.aiia/github-app.json` | `0600` | `app_id`, `installation_id`, `private_key_path`, optional `repositories` |
| `~/.aiia/github-app.pem` | `0600` | App private key (PKCS#1 or PKCS#8 PEM) |

Env vars override the JSON file field-by-field. If ids are in JSON and
`~/.aiia/github-app.pem` exists, the default PEM path is used.

The desktop console's `~/.aiia/keys.json` is a **different** store (cloud
LLM provider keys). Do not put the GitHub App PEM there.

## Status probe

`GET /api/agents/resources` → `github`:

```json
{
  "status": "connected",
  "mode": "read_only",
  "source": "github_app",
  "cli_user_present": true,
  "detail": "read-only GitHub App",
  "app_id": "123456",
  "installation_id": "789012"
}
```

| `status` | Meaning |
|---|---|
| `connected` | App id + installation id + loadable PEM are present |
| `not_configured` | Any required piece is missing |
| `disconnected` | Pieces are present but the PEM does not load |

`cli_user_present` is `true` when `gh auth status` exits 0. It is
**diagnostic only**. A human CLI session cannot make `status` become
`connected`.

Agents with the GitHub read tool get a system-prompt fragment that matches
this probe: when connected, read-only scope and the `gh` ban; otherwise
forbid-claim language.

## Non-goals

- **Do not** point agents at `gh auth token`, `GH_TOKEN`, `GITHUB_TOKEN`,
  or `~/.config/gh/hosts.yml`.
- **Do not** grant write scopes "just in case".
- **Do not** use this App for Command Center tasks that already shell out
  to the operator's `gh` (those are a separate, human-identity path).
- Full Agent Studio tool-loop wiring (the model invoking GET helpers
  mid-run) is follow-up. `local_brain.github_app.GitHubReadClient` is the
  sanctioned stub: App JWT → installation token → GET/HEAD only.

## Verify

```bash
# After placing credentials on the Mini:
curl -s localhost:8200/api/agents/resources | jq .github
# Expect status=connected, source=github_app
# If only `gh auth status` works and no App files exist:
# Expect status=not_configured, cli_user_present=true

pytest local_brain/tests/test_github_app.py
```
