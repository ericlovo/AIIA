# OpenClaw Configuration for AIIA Local Brain

## Provider

Use Anthropic Claude as the reasoning engine for OpenClaw itself.
The local Ollama models are used by the AIIA skills, not by OpenClaw's brain.

```
provider: anthropic
model: claude-sonnet-4-5-20250929
```

## Skills Directory

```
skills_dir: ~/.aiia/AIIA/local_brain/pilot/openclaw_skills/
```

## Environment Variables

These should be set in the shell or in ~/.zshrc:

```
LOCAL_BRAIN_URL=http://localhost:8100
LOCAL_LLM_URL=http://localhost:11434
```

## Security

```
exec.ask: "on"          # Require approval before write/exec commands
sandbox: true           # Run in Docker container
```

## Proactive Tasks

```
schedule:
  - every: 30m
    skill: aiia-brain-health
    notify_if: issues.length > 0
```

## Messaging

Configure during `openclaw onboard`:
- Signal (recommended — encrypted, private)
- WhatsApp (convenient)
- Telegram (easy bot setup)

Choose whichever you use most. The agent will be reachable there 24/7.
