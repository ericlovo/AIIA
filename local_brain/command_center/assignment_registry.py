"""Durable assignments and agent-to-agent handoffs for Agent Studio."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ASSIGNMENT_DATA_FILE = Path(__file__).parent / "assignment_data.json"
MAX_ASSIGNMENTS = 250
MAX_HANDOFFS = 250
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
VALID_ARTIFACT_TYPES = {"brief", "analysis", "plan", "decision", "review"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssignmentRegistry:
    def __init__(self, data_file: Path | None = None):
        self.data_file = data_file or ASSIGNMENT_DATA_FILE
        self.assignments: list[dict[str, Any]] = []
        self.handoffs: list[dict[str, Any]] = []
        self.load()

    def list_assignments(self) -> list[dict[str, Any]]:
        return self.assignments

    def list_handoffs(self) -> list[dict[str, Any]]:
        return self.handoffs

    def get_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        return next(
            (assignment for assignment in self.assignments if assignment["id"] == assignment_id),
            None,
        )

    def get_handoff(self, handoff_id: str) -> dict[str, Any] | None:
        return next(
            (handoff for handoff in self.handoffs if handoff["id"] == handoff_id),
            None,
        )

    def create_assignment(
        self,
        *,
        title: str,
        objective: str,
        agent_id: str,
        priority: str = "normal",
        context: str = "",
        success_criteria: str = "",
        source_handoff_id: str = "",
        assignment_id: str | None = None,
    ) -> dict[str, Any]:
        priority = priority.strip().lower()
        if priority not in VALID_PRIORITIES:
            raise ValueError("invalid_priority")
        now = _now()
        assignment = {
            "id": assignment_id or f"asg_{uuid.uuid4().hex[:12]}",
            "title": title.strip()[:120],
            "objective": objective.strip()[:8_000],
            "agent_id": agent_id.strip()[:80],
            "priority": priority,
            "context": context.strip()[:20_000],
            "success_criteria": success_criteria.strip()[:4_000],
            "source_handoff_id": source_handoff_id,
            "status": "queued",
            "result": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
        self.assignments.insert(0, assignment)
        self.assignments = self.assignments[:MAX_ASSIGNMENTS]
        self.save()
        return assignment

    def set_running(self, assignment_id: str) -> dict[str, Any] | None:
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            return None
        now = _now()
        assignment["status"] = "running"
        assignment["result"] = ""
        assignment["error"] = ""
        assignment["started_at"] = now
        assignment["completed_at"] = None
        assignment["updated_at"] = now
        self._sync_handoff_status(assignment, "running")
        self.save()
        return assignment

    def finish_assignment(
        self, assignment_id: str, *, result: str = "", error: str = ""
    ) -> dict[str, Any] | None:
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            return None
        now = _now()
        status = "failed" if error else "completed"
        assignment["status"] = status
        assignment["result"] = result.strip()[:40_000]
        assignment["error"] = error.strip()[:2_000]
        assignment["updated_at"] = now
        assignment["completed_at"] = now
        self._sync_handoff_status(assignment, status)
        self.save()
        return assignment

    def create_handoff(
        self,
        *,
        source_assignment_id: str,
        to_agent_id: str,
        artifact_type: str,
        instructions: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source = self.get_assignment(source_assignment_id)
        if not source:
            raise ValueError("source_assignment_not_found")
        if source["status"] != "completed" or not source["result"]:
            raise ValueError("source_assignment_not_completed")
        if source["agent_id"] == to_agent_id:
            raise ValueError("handoff_requires_different_agent")

        artifact_type = artifact_type.strip().lower()
        if artifact_type not in VALID_ARTIFACT_TYPES:
            raise ValueError("invalid_artifact_type")

        handoff_id = f"hof_{uuid.uuid4().hex[:12]}"
        assignment_id = f"asg_{uuid.uuid4().hex[:12]}"
        now = _now()
        handoff = {
            "id": handoff_id,
            "source_assignment_id": source_assignment_id,
            "target_assignment_id": assignment_id,
            "from_agent_id": source["agent_id"],
            "to_agent_id": to_agent_id.strip()[:80],
            "artifact_type": artifact_type,
            "artifact": source["result"][:40_000],
            "instructions": instructions.strip()[:8_000],
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
        self.handoffs.insert(0, handoff)
        self.handoffs = self.handoffs[:MAX_HANDOFFS]

        context = (
            f"{artifact_type.title()} handed off from assignment "
            f"'{source['title']}':\n\n{source['result']}"
        )
        target = self.create_assignment(
            title=f"Continue: {source['title']}",
            objective=instructions,
            agent_id=to_agent_id,
            priority=source["priority"],
            context=context,
            success_criteria=source["success_criteria"],
            source_handoff_id=handoff_id,
            assignment_id=assignment_id,
        )
        return handoff, target

    def delete_assignment(self, assignment_id: str) -> bool:
        assignment = self.get_assignment(assignment_id)
        if not assignment or assignment["status"] == "running":
            return False
        if self.assignment_has_handoffs(assignment_id):
            return False
        self.assignments.remove(assignment)
        self.save()
        return True

    def assignment_has_handoffs(self, assignment_id: str) -> bool:
        return any(
            handoff["source_assignment_id"] == assignment_id
            or handoff["target_assignment_id"] == assignment_id
            for handoff in self.handoffs
        )

    def delete_handoff(self, handoff_id: str) -> bool:
        handoff = self.get_handoff(handoff_id)
        if not handoff or handoff["status"] == "running":
            return False
        target = self.get_assignment(handoff["target_assignment_id"])
        if target:
            target["source_handoff_id"] = ""
            target["updated_at"] = _now()
        self.handoffs.remove(handoff)
        self.save()
        return True

    def _sync_handoff_status(self, assignment: dict[str, Any], status: str) -> None:
        handoff_id = assignment.get("source_handoff_id")
        if not handoff_id:
            return
        handoff = self.get_handoff(handoff_id)
        if handoff:
            handoff["status"] = status
            handoff["updated_at"] = _now()

    def save(self) -> None:
        try:
            payload = {
                "assignments": self.assignments,
                "handoffs": self.handoffs,
            }
            self.data_file.write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            logger.error("Could not save assignments: %s", exc)

    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            payload = json.loads(self.data_file.read_text())
            self.assignments = payload.get("assignments", [])[:MAX_ASSIGNMENTS]
            self.handoffs = payload.get("handoffs", [])[:MAX_HANDOFFS]
            interrupted_ids = set()
            for assignment in self.assignments:
                if assignment.get("status") == "running":
                    interrupted_ids.add(assignment["id"])
                    assignment["status"] = "failed"
                    assignment["error"] = "interrupted_by_restart"
                    assignment["updated_at"] = _now()
                    assignment["completed_at"] = assignment["updated_at"]
            for handoff in self.handoffs:
                if handoff.get("target_assignment_id") in interrupted_ids:
                    handoff["status"] = "failed"
                    handoff["updated_at"] = _now()
            if interrupted_ids:
                self.save()
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load assignments: %s", exc)
