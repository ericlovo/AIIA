"""One-line job description and a single output channel for Studio agents.

Read paths derive a one-liner and default channel without rewriting stored
records. Slack is a declared destination only: delivery uses the existing
outbound configured() gates, never a new Slack client, scope, or egress tool.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from local_brain.command_center import slack_memory_posts

ONE_LINER_MAX = 120
USE_WHEN_MAX = ONE_LINER_MAX
OUTPUT_CHANNELS = ("studio_inbox", "slack")
DEFAULT_OUTPUT_CHANNEL = "studio_inbox"
AGENT_KINDS = ("coding", "product", "ops")
CODING_TOOLS = frozenset({"Repository read", "GitHub read", "Git workspace"})
MAX_HANDLES = 12
HANDLE_MAX = 40
VALUE_WINDOW_DAYS = 14
SLACK_NOT_CONFIGURED = "slack not configured"
SLACK_POSTING_PENDING = "slack declared; posting not implemented, delivered to Studio inbox"
RUN_FAILED_NOTE = "not delivered: run failed"
INBOX_UNAVAILABLE_NOTE = "studio inbox unavailable; result kept on the run only"

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def derive_one_liner(mission: str, *, limit: int = ONE_LINER_MAX) -> str:
    """First sentence of the mission, truncated. Empty mission yields empty."""
    text = " ".join(str(mission or "").split())
    if not text:
        return ""
    first = _SENTENCE.split(text, maxsplit=1)[0].strip()
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def stored_one_liner(value: Any) -> str:
    return str(value or "").strip()[:ONE_LINER_MAX]


def stored_use_when(value: Any) -> str:
    return str(value or "").strip()[:USE_WHEN_MAX]


def resolve_one_liner(agent: dict[str, Any]) -> str:
    stored = stored_one_liner(agent.get("one_liner"))
    return stored or derive_one_liner(str(agent.get("mission") or ""))


def normalize_kind(value: Any) -> str:
    """Validate a write. Empty means unset; unknown values are refused."""
    if value is None or value == "":
        return ""
    kind = str(value).strip().lower()
    if kind not in AGENT_KINDS:
        raise ValueError("unknown_agent_kind")
    return kind


def stored_kind(agent: dict[str, Any]) -> str:
    raw = str(agent.get("kind") or "").strip().lower()
    return raw if raw in AGENT_KINDS else ""


def derive_kind(agent: dict[str, Any]) -> str:
    """Best-effort kind when none is stored. Repo or coding tools → coding."""
    tools = {str(tool) for tool in (agent.get("tools") or [])}
    if tools & CODING_TOOLS or str(agent.get("repo_id") or "").strip():
        return "coding"
    return ""


def resolve_kind(agent: dict[str, Any]) -> str:
    return stored_kind(agent) or derive_kind(agent)


def resolve_use_when(agent: dict[str, Any]) -> str:
    stored = stored_use_when(agent.get("use_when"))
    return stored or resolve_one_liner(agent)


def normalize_handles(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("invalid_handles")
    return [str(tag).strip()[:HANDLE_MAX] for tag in value if str(tag).strip()][:MAX_HANDLES]


def present_handles(agent: dict[str, Any]) -> list[str]:
    raw = agent.get("handles")
    if not isinstance(raw, list):
        return []
    try:
        return normalize_handles(raw)
    except ValueError:
        return []


def stored_retired(agent: dict[str, Any]) -> bool:
    return bool(agent.get("retired"))


def normalize_output_channel(value: Any) -> str:
    if value is None or value == "":
        return DEFAULT_OUTPUT_CHANNEL
    channel = str(value).strip()
    if channel not in OUTPUT_CHANNELS:
        raise ValueError("unknown_output_channel")
    return channel


def declared_output_channel(agent: dict[str, Any]) -> str:
    try:
        return normalize_output_channel(agent.get("output_channel"))
    except ValueError:
        return DEFAULT_OUTPUT_CHANNEL


def slack_outbound_configured() -> bool:
    """True when the Slack memory-post outbound path is configured.

    Only memory posts carry content to an allowlisted channel. The receipt
    gate proves a bot token exists for acks, not that an agent-output
    destination does, so it is not consulted here. No new scope or AIRGAP
    exception is involved either way.
    """
    return slack_memory_posts.configured()


def output_channel_note(agent: dict[str, Any], *, slack_ready: bool | None = None) -> str:
    if declared_output_channel(agent) != "slack":
        return ""
    ready = slack_outbound_configured() if slack_ready is None else slack_ready
    return SLACK_POSTING_PENDING if ready else SLACK_NOT_CONFIGURED


def resolve_delivery(agent: dict[str, Any], *, slack_ready: bool | None = None) -> dict[str, str]:
    """Pick the one channel this run is delivered to.

    Every run is delivered to the Studio inbox today. Slack is a declared
    destination only: nothing posts agent output to Slack yet, so recording
    ``delivered_channel="slack"`` would describe output that went nowhere.
    The note says why the declared channel was not used. Later digest work
    reuses slack_memory_posts and can flip this when a real post exists.
    """
    declared = declared_output_channel(agent)
    note = output_channel_note(agent, slack_ready=slack_ready)
    return {"declared": declared, "delivered_channel": DEFAULT_OUTPUT_CHANNEL, "note": note}


def present_agent(
    agent: dict[str, Any],
    *,
    slack_ready: bool | None = None,
    value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy for API/UI. Derived fields are not written back to disk."""
    shown = dict(agent)
    stored = stored_one_liner(agent.get("one_liner"))
    shown["one_liner"] = stored or derive_one_liner(str(agent.get("mission") or ""))
    shown["one_liner_derived"] = not bool(stored)
    stored_when = stored_use_when(agent.get("use_when"))
    shown["use_when"] = stored_when or shown["one_liner"]
    shown["use_when_derived"] = not bool(stored_when)
    stored_k = stored_kind(agent)
    shown["kind"] = stored_k or derive_kind(agent)
    shown["kind_derived"] = not bool(stored_k)
    shown["retired"] = stored_retired(agent)
    shown["handles"] = present_handles(agent)
    shown["output_channel"] = declared_output_channel(agent)
    shown["output_channel_note"] = output_channel_note(agent, slack_ready=slack_ready)
    if value is not None:
        shown["value"] = value
    return shown


def _in_window(stamp: str | None, start: datetime) -> bool:
    if not stamp:
        return False
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when >= start


def agent_value(
    agent: dict[str, Any],
    assignments: list[dict[str, Any]],
    *,
    runs_in_window: int = 0,
    now: datetime | None = None,
    window_days: int = VALUE_WINDOW_DAYS,
) -> dict[str, Any]:
    """Runs, last run, and reviewed vs unreviewed outputs in the window."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=window_days)
    reviewed = 0
    unreviewed = 0
    agent_id = agent.get("id")
    for item in assignments:
        if item.get("agent_id") != agent_id:
            continue
        if item.get("status") != "completed" or not str(item.get("result") or "").strip():
            continue
        stamp = item.get("reviewed_at") or item.get("completed_at") or item.get("created_at")
        if not _in_window(stamp, start):
            continue
        if item.get("review_status") in {"accepted", "rejected"}:
            reviewed += 1
        elif item.get("review_status", "unreviewed") == "unreviewed" and not item.get(
            "dismissed_at"
        ):
            unreviewed += 1
    return {
        "window_days": window_days,
        "runs": int(runs_in_window),
        "last_run_at": agent.get("last_run_at"),
        "reviewed": reviewed,
        "unreviewed": unreviewed,
    }


def runs_in_window_by_agent(
    activity: dict[str, Any] | None,
    *,
    window_days: int = VALUE_WINDOW_DAYS,
    now: datetime | None = None,
) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    start = (now - timedelta(days=window_days - 1)).date().isoformat()
    counts: dict[str, int] = {}
    for row in (activity or {}).get("agent_days") or []:
        day = str(row.get("day") or "")
        agent_id = str(row.get("agent_id") or "")
        if not agent_id or day < start:
            continue
        counts[agent_id] = counts.get(agent_id, 0) + int(row.get("total") or 0)
    return counts


def present_agents(
    agents: list[dict[str, Any]],
    assignments: list[dict[str, Any]] | None = None,
    activity: dict[str, Any] | None = None,
    *,
    slack_ready: bool | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    ready = slack_outbound_configured() if slack_ready is None else slack_ready
    work = assignments or []
    run_counts = runs_in_window_by_agent(activity, now=now)
    return [
        present_agent(
            agent,
            slack_ready=ready,
            value=agent_value(
                agent, work, runs_in_window=run_counts.get(agent.get("id"), 0), now=now
            ),
        )
        for agent in agents
    ]


def inbox_title(agent: dict[str, Any], task: str) -> str:
    """Lead with the task so repeated manual runs of one agent stay distinguishable."""
    label = str(agent.get("name") or "").strip() or resolve_one_liner(agent) or "Agent"
    task_line = " ".join(str(task or "").split())
    if not task_line:
        return label[:120]
    room = 120 - len(label) - 3
    if room < 24:
        return task_line[:120]
    task_part = task_line if len(task_line) <= room else task_line[: room - 1].rstrip() + "…"
    return f"{task_part} — {label}"
