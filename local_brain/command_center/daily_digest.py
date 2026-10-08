"""The daily digest: one line per agent and per loop, from records, with no model.

Spec item 2 of the Studio build-out: "surface one line per day: what moved,
what's stuck." The digest is the only thing the loops should put in front of a
person every day; everything else they file is a proposal that waits for a
decision. Delivery reuses the memory-post outbox, so it reaches the allowlisted
Slack channel when that is configured and is otherwise one inbox row.
"""

import json
import os
from pathlib import Path

MAX_BODY = 2_800
VERDICT_CHARS = 80
NOTE_CHARS = 90


def loops_registry_path() -> Path:
    override = os.getenv("AIIA_LOOPS_REGISTRY", "")
    return Path(override) if override else Path.home() / ".aiia" / "loops-registry.json"


def load_loops(path: Path | None = None) -> dict:
    target = path or loops_registry_path()
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def digest_key(date: str) -> str:
    return f"digest:{date}"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _first_line(text) -> str:
    for line in str(text or "").splitlines():
        line = line.strip().lstrip("#*- ").strip()
        if line:
            return line[:VERDICT_CHARS]
    return ""


def _agent_line(agent: dict, assignments: list[dict], runs: int, date: str) -> str:
    agent_id = agent.get("id")
    waiting = failed = 0
    for item in assignments:
        if item.get("agent_id") != agent_id:
            continue
        stamp = str(item.get("completed_at") or item.get("updated_at") or "")
        if item.get("status") == "failed" and stamp[:10] == date:
            failed += 1
        elif (
            item.get("status") == "completed"
            and item.get("review_status", "unreviewed") == "unreviewed"
            and not item.get("dismissed_at")
            and str(item.get("result") or "").strip()
        ):
            waiting += 1
    parts = [f"{runs} run{'s' if runs != 1 else ''}"]
    if waiting:
        parts.append(f"{waiting} waiting review")
    if failed:
        parts.append(f"{failed} failed")
    if not agent.get("loop_enabled"):
        parts.append("no schedule")
    line = f"- {agent.get('name') or agent_id}: " + ", ".join(parts)
    if str(agent.get("last_run_at") or "")[:10] == date:
        verdict = _first_line(agent.get("last_error") or agent.get("last_result"))
        if verdict:
            line += f" — {verdict}"
    return line


def build_digest(
    *,
    date: str,
    agents: list[dict],
    assignments: list[dict],
    run_counts: dict[str, int],
    loops: dict,
    tasks: list[dict],
    inbox_counts: dict[str, int],
) -> str:
    lines = [f"AIIA digest {date}", "", "Agents"]
    for agent in sorted(agents, key=lambda a: str(a.get("name") or "")):
        lines.append(_agent_line(agent, assignments, int(run_counts.get(agent.get("id"), 0)), date))
    if not agents:
        lines.append("- no agents")
    lines += ["", "Loops"]
    for name, entry in sorted(loops.items()):
        if not isinstance(entry, dict):
            continue
        last = str(entry.get("last_run") or "")[:16].replace("T", " ")
        status = entry.get("last_status") or "never run"
        note = str(entry.get("last_note") or "").strip()[:NOTE_CHARS]
        lines.append(f"- {name}: {status} {last}".rstrip() + (f" — {note}" if note else ""))
    if not loops:
        lines.append("- no loop registry found")
    failing = [t for t in tasks if t.get("last_status") == "failed"]
    if failing:
        lines += ["", "Built-in tasks failing"]
        for task in failing:
            reason = _first_line(str(task.get("last_result") or "").replace("FAILED: ", ""))
            lines.append(
                f"- {task.get('name') or task.get('task_id')}" + (f": {reason}" if reason else "")
            )
    waiting = sum(inbox_counts.values())
    if waiting:
        detail = ", ".join(f"{count} {source}" for source, count in sorted(inbox_counts.items()))
        lines += ["", f"Inbox waiting review: {waiting} ({detail})"]
    else:
        lines += ["", "Inbox waiting review: 0"]
    body = "\n".join(lines)
    if len(body) > MAX_BODY:
        body = body[: MAX_BODY - 1].rstrip() + "…"
    return body
