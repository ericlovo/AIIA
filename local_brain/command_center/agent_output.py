"""One-line job description and a single output channel for Studio agents.

Read paths derive a one-liner and default channel without rewriting stored
records. Slack is a declared destination only: delivery uses the existing
outbound configured() gates, never a new Slack client, scope, or egress tool.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from local_brain.command_center import slack_memory_posts, slack_receipts

ONE_LINER_MAX = 120
OUTPUT_CHANNELS = ("studio_inbox", "slack")
DEFAULT_OUTPUT_CHANNEL = "studio_inbox"
VALUE_WINDOW_DAYS = 14
SLACK_NOT_CONFIGURED = "slack not configured"

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


def resolve_one_liner(agent: dict[str, Any]) -> str:
    stored = stored_one_liner(agent.get("one_liner"))
    return stored or derive_one_liner(str(agent.get("mission") or ""))


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
    """True when an existing Slack outbound path can send.

    Memory posts carry content to the allowlisted channel. Receipts share the
    same bot token and post_message transport. Either gate means Slack egress
    is already wired; neither adds a new scope or AIRGAP exception.
    """
    return slack_memory_posts.configured() or slack_receipts.configured()


def output_channel_note(agent: dict[str, Any], *, slack_ready: bool | None = None) -> str:
    if declared_output_channel(agent) != "slack":
        return ""
    ready = slack_outbound_configured() if slack_ready is None else slack_ready
    return "" if ready else SLACK_NOT_CONFIGURED


def resolve_delivery(agent: dict[str, Any], *, slack_ready: bool | None = None) -> dict[str, str]:
    """Pick the one channel this run is delivered to.

    Slack stays a declared destination. If the existing outbound path is not
    configured, the result falls back to the Studio inbox and the agent shows
    a slack-not-configured note. No new Slack post is sent here; later digest
    work reuses slack_memory_posts / slack_receipts.
    """
    declared = declared_output_channel(agent)
    ready = slack_outbound_configured() if slack_ready is None else slack_ready
    if declared == "slack" and ready:
        return {"declared": declared, "delivered_channel": "slack", "note": ""}
    if declared == "slack":
        return {
            "declared": declared,
            "delivered_channel": DEFAULT_OUTPUT_CHANNEL,
            "note": SLACK_NOT_CONFIGURED,
        }
    return {"declared": declared, "delivered_channel": DEFAULT_OUTPUT_CHANNEL, "note": ""}


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
    label = resolve_one_liner(agent) or str(agent.get("name") or "Agent").strip() or "Agent"
    task_line = " ".join(str(task or "").split())
    title = f"{label}: {task_line}" if task_line else label
    return title[:120]
