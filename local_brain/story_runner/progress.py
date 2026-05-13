"""
Progress tracking for AIIA Story Runner.

Manages action_plan.json — the source of truth for story execution.
Each session reads it fresh, enabling pause/resume across sessions.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActionPlan:
    """
    Persistent plan for a story execution.

    Written by the Planner agent, consumed by the Coder agent.
    Stored as action_plan.json in the worktree root.
    """

    story: str = ""
    product: str = ""
    branch: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    planner_session_id: str | None = None
    planner_cost: float | None = None
    total_cost: float | None = 0.0
    created_at: str | None = None
    updated_at: str | None = None

    def next_action(self) -> dict[str, Any] | None:
        """Get the next incomplete action."""
        for action in self.actions:
            if not action.get("completed", False):
                return action
        return None

    def mark_complete(self, index: int) -> None:
        """Mark an action as completed."""
        if 0 <= index < len(self.actions):
            self.actions[index]["completed"] = True

    @property
    def progress(self) -> str:
        """Human-readable progress string."""
        completed = sum(1 for a in self.actions if a.get("completed", False))
        total = len(self.actions)
        return f"{completed}/{total}"

    @property
    def is_complete(self) -> bool:
        """True when all actions are done."""
        return all(a.get("completed", False) for a in self.actions)


def load_plan(path: Path) -> ActionPlan:
    """Load an action plan from JSON."""
    data = json.loads(path.read_text())
    return ActionPlan(
        story=data.get("story", ""),
        product=data.get("product", ""),
        branch=data.get("branch", ""),
        actions=data.get("actions", []),
        planner_session_id=data.get("planner_session_id"),
        planner_cost=data.get("planner_cost"),
        total_cost=data.get("total_cost", 0.0),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def save_plan(plan: ActionPlan, path: Path) -> None:
    """Save an action plan to JSON."""
    from datetime import datetime

    plan.updated_at = datetime.now().isoformat()
    path.write_text(json.dumps(asdict(plan), indent=2, default=str) + "\n")
