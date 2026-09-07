"""Voice Conductor — bounded Grok Voice tools over Agent Studio.

Grok is the speech + reasoning plane only. Every tool either reads Command
Center state or creates/runs an Assignment. Forbidden capabilities (shell,
git push/open_pr, unrestricted writes, Sanction spend) are fail-closed.

Key loading follows the Mini secret pattern: ``XAI_API_KEY`` env first, then
``~/.aiia/keys.json``. The long-lived key never leaves this process.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiia.voice_conductor")

XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
XAI_CLIENT_SECRETS_URL = "https://api.x.ai/v1/realtime/client_secrets"
XAI_VOICE = "eve"
XAI_MODEL = "grok-voice-latest"
EPHEMERAL_TTL_SECONDS = 300
EGRESS_TOOL = "xai.realtime"

# Read tools — Command Center / Agent Studio state only.
READ_TOOLS = frozenset(
    {
        "list_agents",
        "list_assignments",
        "list_handoffs",
        "list_resources",
        "mini_status",
    }
)

# Act tools — create or run Assignments. Never shell, git write, or spend.
ACT_TOOLS = frozenset(
    {
        "create_assignment",
        "run_assignment",
    }
)

ALLOWED_TOOLS = READ_TOOLS | ACT_TOOLS

# Names Grok (or a tampered client) might invent. Always rejected.
FORBIDDEN_TOOLS = frozenset(
    {
        "run_shell",
        "shell",
        "execute_command",
        "git_push",
        "push",
        "open_pr",
        "git_write",
        "write_file",
        "approve_git_write",
        "approve_git_workspace",
        "create_handoff",
        "delete_agent",
        "delete_assignment",
        "sanction_spend",
        "authorize_spend",
        "gh_token",
        "web_search",
        "x_search",
        "file_search",
        "mcp",
    }
)

VALID_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})

VOICE_CONDUCTOR_SPECIALTY = {
    "slug": "voice-conductor",
    "version": "0.1.0",
    "name": "Voice Conductor",
    "depth": 2,
    "recommendedDepth": 2,
    "maxDelegationDepth": 0,
    "authority": (
        "Orchestrator that creates and retries Assignments for existing "
        "Agent Studio agents. Not a D4 executive: no budgets, no Specialty "
        "installs, no Handoff edges, no git push/open_pr."
    ),
    "requiredTools": sorted(ALLOWED_TOOLS),
    "producedArtifacts": ["assignment"],
    "requiredApprovalLevel": "supervised",
}

VOICE_CONDUCTOR_INSTRUCTIONS = """You are the AIIA Voice Conductor, a D2 orchestrator.

You speak to Eric. You do not execute work yourself. You create and run typed
Assignments for existing Agent Studio agents on the Mini. The Mini executes
privately; you are the voice interface over that org graph.

Rules:
- Use only the provided tools. Never invent shell, git, spend, or file-write tools.
- Before creating an Assignment, list_agents and pick an existing agent_id.
- create_assignment needs a short title, a concrete objective, and agent_id.
- run_assignment only works on queued or failed Assignments. If mini_status
  says busy, tell Eric the Mini is occupied and leave the Assignment queued.
- Read list_assignments / list_handoffs / list_resources when asked for status.
- Do not claim you pushed code, opened a PR, spent Sanction budget, or fired
  a Handoff. Those are out of scope for voice.
- Be brief. Confirm what you created or ran. Ask before assigning urgent work.
"""


@dataclass
class VoiceConductorDeps:
    """Injected Command Center registries. Tests pass fakes; production wires server.py."""

    list_agents: Callable[[], list[dict[str, Any]]]
    get_agent: Callable[[str], dict[str, Any] | None]
    list_assignments: Callable[[], list[dict[str, Any]]]
    get_assignment: Callable[[str], dict[str, Any] | None]
    create_assignment: Callable[..., dict[str, Any]]
    list_handoffs: Callable[[], list[dict[str, Any]]]
    github_status: Callable[[], dict[str, Any]]
    available_repos: Callable[[], list[dict[str, Any]]]
    mini_busy: Callable[[], bool]
    run_assignment: Callable[[str], Awaitable[dict[str, Any]]] | None = None


@dataclass
class ToolResult:
    ok: bool
    name: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    status: int = 200

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "name": self.name}
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = self.error
        return payload


def keys_json_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".aiia" / "keys.json"


def load_xai_api_key(*, home: Path | None = None, environ: dict[str, str] | None = None) -> str:
    """Return the long-lived xAI key or '' if unset. Never log the value."""
    env = environ if environ is not None else os.environ
    from_env = (env.get("XAI_API_KEY") or "").strip()
    if from_env:
        return from_env

    path = keys_json_path(home)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""
    if not isinstance(data, dict):
        return ""

    for key in ("XAI_API_KEY", "xai_api_key", "xai"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    providers = data.get("providers")
    if isinstance(providers, dict):
        value = providers.get("xai")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def voice_configured(*, home: Path | None = None, environ: dict[str, str] | None = None) -> bool:
    return bool(load_xai_api_key(home=home, environ=environ))


def airgap_blocks_voice() -> bool:
    return os.getenv("AIIA_AIRGAP", "").strip().lower() in {"1", "true", "yes"}


def status_payload(
    *, home: Path | None = None, environ: dict[str, str] | None = None
) -> dict[str, Any]:
    configured = voice_configured(home=home, environ=environ)
    if airgap_blocks_voice():
        reason = "airgap"
        state = "not_configured"
    elif not configured:
        reason = "missing_xai_api_key"
        state = "not_configured"
    else:
        reason = ""
        state = "connected"
    return {
        "status": state,
        "configured": configured and state == "connected",
        "provider": "xai",
        "model": XAI_MODEL,
        "voice": XAI_VOICE,
        "realtime_url": XAI_REALTIME_URL,
        "reason": reason,
        "tools": grok_tool_definitions(),
        "allowlist": sorted(ALLOWED_TOOLS),
        "forbidden": sorted(FORBIDDEN_TOOLS),
        "specialty": VOICE_CONDUCTOR_SPECIALTY,
    }


def grok_tool_definitions() -> list[dict[str, Any]]:
    """Function tools advertised to Grok. Server-side xAI tools are omitted."""
    return [
        {
            "type": "function",
            "name": "list_agents",
            "description": "List Agent Studio agents (id, name, mission, status).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "list_assignments",
            "description": "List Assignments (id, title, agent_id, status, priority).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "list_handoffs",
            "description": "List typed Handoffs between agents. Read-only.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "list_resources",
            "description": "Repository mounts and GitHub connection status (read-only).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "mini_status",
            "description": "Whether the Mini is busy (one Run at a time).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "create_assignment",
            "description": (
                "Create a queued Assignment for an existing Agent Studio agent. "
                "Does not start the Mini."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title, max 120 chars."},
                    "objective": {
                        "type": "string",
                        "description": "What the agent should produce.",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Existing agent id from list_agents.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                    },
                    "context": {"type": "string"},
                    "success_criteria": {"type": "string"},
                },
                "required": ["title", "objective", "agent_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "run_assignment",
            "description": (
                "Run or retry a queued or failed Assignment on the Mini. "
                "Returns mini_busy if the Mini is already running a job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignment_id": {"type": "string"},
                },
                "required": ["assignment_id"],
                "additionalProperties": False,
            },
        },
    ]


def session_config() -> dict[str, Any]:
    """session.update payload. Push-to-talk: no server VAD; client commits."""
    return {
        "voice": XAI_VOICE,
        "instructions": VOICE_CONDUCTOR_INSTRUCTIONS,
        "turn_detection": None,
        "tools": grok_tool_definitions(),
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "grok-transcribe", "language_hint": "en"},
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
    }


def classify_tool(name: str) -> str:
    """Return 'allowed', 'forbidden', or 'unknown'."""
    cleaned = (name or "").strip()
    if cleaned in FORBIDDEN_TOOLS:
        return "forbidden"
    if cleaned in ALLOWED_TOOLS:
        return "allowed"
    return "unknown"


def _public_agent(agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": agent.get("id"),
        "name": agent.get("name"),
        "mission": agent.get("mission"),
        "status": agent.get("status"),
        "skills": agent.get("skills", []),
        "tools": agent.get("tools", []),
        "last_run_at": agent.get("last_run_at"),
    }


def _public_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": assignment.get("id"),
        "title": assignment.get("title"),
        "objective": assignment.get("objective"),
        "agent_id": assignment.get("agent_id"),
        "priority": assignment.get("priority"),
        "status": assignment.get("status"),
        "error": assignment.get("error"),
        "created_at": assignment.get("created_at"),
        "updated_at": assignment.get("updated_at"),
    }


def _public_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": handoff.get("id"),
        "source_assignment_id": handoff.get("source_assignment_id"),
        "target_assignment_id": handoff.get("target_assignment_id"),
        "from_agent_id": handoff.get("from_agent_id"),
        "to_agent_id": handoff.get("to_agent_id"),
        "artifact_type": handoff.get("artifact_type"),
        "status": handoff.get("status"),
        "created_at": handoff.get("created_at"),
    }


async def execute_tool(
    name: str,
    arguments: dict[str, Any] | None,
    deps: VoiceConductorDeps,
) -> ToolResult:
    """Fail-closed tool dispatcher. Unknown and forbidden names never run."""
    kind = classify_tool(name)
    if kind == "forbidden":
        return ToolResult(ok=False, name=name, error="tool_forbidden", status=403)
    if kind != "allowed":
        return ToolResult(
            ok=False, name=name or "unknown", error="tool_not_allowlisted", status=403
        )

    args = arguments if isinstance(arguments, dict) else {}
    try:
        if name == "list_agents":
            agents = [_public_agent(agent) for agent in deps.list_agents()]
            return ToolResult(ok=True, name=name, result={"agents": agents, "count": len(agents)})
        if name == "list_assignments":
            items = [_public_assignment(item) for item in deps.list_assignments()]
            return ToolResult(
                ok=True, name=name, result={"assignments": items, "count": len(items)}
            )
        if name == "list_handoffs":
            items = [_public_handoff(item) for item in deps.list_handoffs()]
            return ToolResult(ok=True, name=name, result={"handoffs": items, "count": len(items)})
        if name == "list_resources":
            return ToolResult(
                ok=True,
                name=name,
                result={"repos": deps.available_repos(), "github": deps.github_status()},
            )
        if name == "mini_status":
            busy = bool(deps.mini_busy())
            return ToolResult(
                ok=True,
                name=name,
                result={"mini_busy": busy, "status": "busy" if busy else "idle"},
            )
        if name == "create_assignment":
            return _create_assignment(args, deps)
        if name == "run_assignment":
            return await _run_assignment(args, deps)
    except ValueError as exc:
        return ToolResult(ok=False, name=name, error=str(exc), status=422)
    except Exception:
        logger.exception("Voice Conductor tool failed: %s", name)
        return ToolResult(ok=False, name=name, error="tool_failed", status=500)
    return ToolResult(ok=False, name=name, error="tool_not_allowlisted", status=403)


def _create_assignment(args: dict[str, Any], deps: VoiceConductorDeps) -> ToolResult:
    title = str(args.get("title") or "").strip()
    objective = str(args.get("objective") or "").strip()
    agent_id = str(args.get("agent_id") or "").strip()
    priority = str(args.get("priority") or "normal").strip().lower() or "normal"
    if not title or not objective or not agent_id:
        return ToolResult(
            ok=False, name="create_assignment", error="missing_required_fields", status=422
        )
    if priority not in VALID_PRIORITIES:
        return ToolResult(ok=False, name="create_assignment", error="invalid_priority", status=422)
    if not deps.get_agent(agent_id):
        return ToolResult(ok=False, name="create_assignment", error="agent_not_found", status=422)
    assignment = deps.create_assignment(
        title=title,
        objective=objective,
        agent_id=agent_id,
        priority=priority,
        context=str(args.get("context") or ""),
        success_criteria=str(args.get("success_criteria") or ""),
    )
    return ToolResult(
        ok=True,
        name="create_assignment",
        result={"assignment": _public_assignment(assignment)},
    )


async def _run_assignment(args: dict[str, Any], deps: VoiceConductorDeps) -> ToolResult:
    assignment_id = str(args.get("assignment_id") or "").strip()
    if not assignment_id:
        return ToolResult(
            ok=False, name="run_assignment", error="missing_assignment_id", status=422
        )
    assignment = deps.get_assignment(assignment_id)
    if not assignment:
        return ToolResult(ok=False, name="run_assignment", error="assignment_not_found", status=404)
    if assignment.get("status") not in {"queued", "failed"}:
        return ToolResult(
            ok=False, name="run_assignment", error="assignment_not_runnable", status=409
        )
    if deps.mini_busy():
        return ToolResult(ok=False, name="run_assignment", error="mini_busy", status=409)
    if deps.run_assignment is None:
        return ToolResult(ok=False, name="run_assignment", error="runner_unavailable", status=503)
    try:
        payload = await deps.run_assignment(assignment_id)
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        status = getattr(exc, "status_code", 500)
        if detail == "mini_busy" or status == 409:
            return ToolResult(
                ok=False, name="run_assignment", error=str(detail or "mini_busy"), status=409
            )
        if status in {404, 409, 422}:
            return ToolResult(
                ok=False, name="run_assignment", error=str(detail or "run_rejected"), status=status
            )
        logger.exception("Voice Conductor run_assignment failed")
        return ToolResult(
            ok=False, name="run_assignment", error="assignment_run_failed", status=500
        )
    public: dict[str, Any] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("assignment"), dict):
            public["assignment"] = _public_assignment(payload["assignment"])
        if isinstance(payload.get("agent"), dict):
            public["agent"] = _public_agent(payload["agent"])
        public["model"] = payload.get("model")
        public["latency_ms"] = payload.get("latency_ms")
    return ToolResult(ok=True, name="run_assignment", result=public)
