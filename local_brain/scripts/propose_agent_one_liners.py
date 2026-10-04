"""Propose one_liner values for existing Studio agents. Read-only.

Prints a unified-style diff of stored vs derived one-liners so a person can
approve them. It never writes agent_data.json, never talks to a live API, and
does not run unless a file is passed.

    python -m local_brain.scripts.propose_agent_one_liners --file /path/to/agents.json

The file may be a registry dump ({"agents": [...]}) or a bare list of agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from local_brain.command_center.agent_output import derive_one_liner, stored_one_liner


def load_agents(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("agents"), list):
        return payload["agents"]
    if isinstance(payload, list):
        return payload
    raise SystemExit("expected an object with an agents list, or a list of agents")


def proposals(agents: list[dict]) -> list[dict]:
    rows = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        stored = stored_one_liner(agent.get("one_liner"))
        proposed = stored or derive_one_liner(str(agent.get("mission") or ""))
        rows.append(
            {
                "id": str(agent.get("id") or ""),
                "name": str(agent.get("name") or ""),
                "stored": stored,
                "proposed": proposed,
                "changed": stored != proposed,
            }
        )
    return rows


def render_diff(path: Path, rows: list[dict]) -> str:
    lines = [
        f"--- {path} (stored one_liner)",
        f"+++ {path} (proposed one_liner from mission)",
        f"@@ {sum(1 for row in rows if row['changed'])} of {len(rows)} agents would change @@",
    ]
    for row in rows:
        label = f"{row['id'] or 'unknown'} {row['name'] or '(unnamed)'}".strip()
        if not row["changed"]:
            lines.append(f" {label}: {row['proposed'] or '(empty)'}")
            continue
        lines.append(f"-{label}: {row['stored'] or '(missing)'}")
        lines.append(f"+{label}: {row['proposed'] or '(empty)'}")
    lines.append("")
    lines.append("No files were written. Copy approved lines into a PATCH/PUT.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to an agents JSON dump. Required so this never touches live data by accident.",
    )
    args = parser.parse_args(argv)
    path = args.file.expanduser()
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    text = render_diff(path, proposals(load_agents(path)))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
