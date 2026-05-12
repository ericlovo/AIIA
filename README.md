# ⌬ AIIA — AI Information Architecture

> A·I·I·A. A palindrome on purpose: the architecture mirrors the name.

Your memory on one side. Your AI tools on the other. AIIA is the surface
in between — where what you decide becomes what they know, and what they
notice becomes what you remember.

Without it, every session starts blank. Every decision gets re-explained.
Every pattern gets re-derived. With it, one continuous thread runs
between your thinking and the tools that act on it. Claude Code, Cursor,
Copilot — pick your tool. The mirror works the same.

Open source. Apache 2.0. Local-first. Free model. Your data stays yours.

[![Version](https://img.shields.io/badge/version-0.5.0-blue)](./CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![Status](https://img.shields.io/badge/status-beta-yellow)](#status)

---

## Quick start

```bash
pip install aiia              # PyPI install
aiia                          # first-run check + setup prompt
aiia next                     # surface your top story to work on now
aiia ask "what did I decide about lockfile policy?"
```

That's the daily loop. `aiia next` chooses, `aiia ask` recalls, `aiia memory add` captures.

If you don't want PyPI, clone and `pip install -e .[dev]`. AIIA needs
Python 3.10+, ~5GB free for a local model, and either Ollama running on
`localhost:11434` or an `ANTHROPIC_API_KEY` if you prefer cloud tiers.

---

## What flows through the mirror

**From you, into your AI tools' working context:**
- Decisions you've made, with the *why* — recoverable, not buried in a chat log
- Patterns and conventions you've named — applied without re-explaining
- Stories in your backlog — surfaced when relevant, ranked by priority × freshness
- Session history — what you were doing when you stopped

**From your AI tools, back into your record:**
- Observations from running code — what changed, what broke, what's pending
- Decisions captured during sessions — saved without you stopping to write them down
- Lessons from debugging — kept for next time
- New stories — when something turns up that deserves its own focused attention

**The substrate that makes both directions work:**
- 9 typed memory categories, quality-scored, decay-aware, hand-editable
- Story prioritization across Obsidian, GitHub Issues, or a JSON file
- Local-first three-tier LLM stack (LOCAL → ANTHROPIC → GOOGLE)
- Background jobs that run while you sleep — security scans, daily reports, memory consolidation
- MCP server so Claude Code talks to AIIA natively
- A CLI (`aiia`) for direct use without going through any other tool

---

## How it works

```
   Your memory ⇄ ⌬ AIIA ⇄ Your AI tools
         │
         ├── Decisions, patterns, stories, sessions persist on your Mac
         ├── Speaks to Claude Code / Cursor / etc. via MCP
         ├── Captures AI observations back into your record
         ├── Free local model by default; Anthropic / Google as fallbacks
         └── CLI surface (`aiia next`, `aiia ask`, `aiia memory`)
```

Two surfaces are running underneath:

- **Brain API** on `localhost:8100` — memory, RAG, LLM routing
- **Command Center** on `localhost:8200` — dashboard, story queue, voice

The CLI is a thin client over both. Claude Code reaches AIIA over MCP.
Your terminal reaches AIIA over `aiia`. Same brain. Both halves of the
mirror in sync.

---

## What AIIA is, that other tools aren't

**vs. Claude Code / Cursor / Copilot directly.** Those are one-directional —
they consume your context, but their observations don't flow back into your
memory. Every session ends and the work is gone. AIIA writes both ways:
your decisions reach their context, their work re-enters your record.

**vs. RAG-only solutions.** RAG is one-directional retrieval — find similar
chunks, return them. AIIA is bidirectional flow — typed memory + scheduled
work + story capture + safety-gated execution. Memory isn't "search this
embedding"; it's structured records with categories, decay, and quality
scoring.

**vs. Custom GPTs / system prompts.** Static one-shot context. AIIA is a
continuous mirror — context updates as your work progresses, in both
directions, without you copying anything anywhere.

**vs. Computer-use agents (OpenClaw, NemoClaw, etc.).** Those operate your
computer. AIIA operates the layer *between* you and your tools — it
maintains memory, schedules background work, and feeds context through
MCP. Infrastructure, not a screen driver.

---

## Who it's for

- Solo developers and small teams running production SaaS who want their
  AI tools to have institutional memory
- Platform engineers building multi-tenant products who need a local
  intelligence layer that knows every service, tenant, and deployment
- Anyone tired of re-explaining context to Claude / Cursor / Copilot
  every single session

---

## Built at Aplora AI

AIIA is open-sourced from the runtime layer powering [Aplora AI](https://aplora.ai),
a multi-tenant AI platform for professional services — law firms,
financial analysts, marketing agencies, healthcare operations. The
substrate that mirrors your developer memory is the same substrate that
mirrors Aplora's tenants' institutional memory: every legal doc reviewed,
every financial pattern observed, every compliance decision made.

If AIIA helps your daily work and you want a managed version with team
sync, audit logs, and SOC 2 / HIPAA posture, [we're building that](https://aplora.ai).
The OSS version is the foundation. Everything load-bearing lives here, in
public, under Apache 2.0.

---

## Status

**v0.5.0 (Q2 2026).** Beta. Running in production on the author's machine
since February 2026. Interfaces stable but evolving — pin a version if
you depend on it.

See [CHANGELOG.md](./CHANGELOG.md) for the release history.
See [SECURITY.md](./SECURITY.md) for vulnerability reporting.
See [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a PR.

---

## License

Apache 2.0. See [LICENSE](./LICENSE).
