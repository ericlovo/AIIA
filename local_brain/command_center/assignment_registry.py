"""Durable assignments and agent-to-agent handoffs for Agent Studio."""

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from local_brain.command_center.persistence import atomic_write_json

logger = logging.getLogger(__name__)

ASSIGNMENT_DATA_FILE = Path(__file__).parent / "assignment_data.json"
MAX_ASSIGNMENTS = 250
MAX_HANDOFFS = 250
MAX_RESULT_LENGTH = 40_000
MAX_CONTEXT_LENGTH = MAX_RESULT_LENGTH + 1_000
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
VALID_ARTIFACT_TYPES = {"brief", "analysis", "plan", "decision", "review"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _durable_mutation(method):
    @wraps(method)
    def mutate(self, *args, **kwargs):
        if self._mutating:
            return method(self, *args, **kwargs)
        assignments, handoffs = list(self.assignments), list(self.handoffs)
        before = deepcopy((assignments, handoffs))
        self._mutating = True
        try:
            result = method(self, *args, **kwargs)
            if (self.assignments, self.handoffs) != before:
                self.save()
            return result
        except Exception:
            # Restore referenced records and list order, including nested handoff creation.
            for records, snapshots in zip((assignments, handoffs), before):
                for record, snapshot in zip(records, snapshots):
                    record.clear()
                    record.update(snapshot)
            self.assignments[:] = assignments
            self.handoffs[:] = handoffs
            raise
        finally:
            self._mutating = False

    return mutate


class AssignmentRegistry:
    def __init__(self, data_file: Path | None = None):
        self.data_file = data_file or ASSIGNMENT_DATA_FILE
        self.assignments: list[dict[str, Any]] = []
        self.handoffs: list[dict[str, Any]] = []
        self._mutating = False
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

    @_durable_mutation
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
        if len(context) > MAX_CONTEXT_LENGTH:
            raise ValueError("assignment_context_too_long")
        self._make_assignment_room()
        now = _now()
        assignment = {
            "id": assignment_id or f"asg_{uuid.uuid4().hex[:12]}",
            "title": title.strip()[:120],
            "objective": objective.strip()[:8_000],
            "agent_id": agent_id.strip()[:80],
            "priority": priority,
            "context": context,
            "success_criteria": success_criteria.strip()[:4_000],
            "source_handoff_id": source_handoff_id,
            "status": "queued",
            "result": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            **self._new_review(),
        }
        self.assignments.insert(0, assignment)
        return assignment

    def _make_assignment_room(self) -> None:
        while len(self.assignments) >= MAX_ASSIGNMENTS:
            disposable = next(
                (
                    assignment
                    for assignment in reversed(self.assignments)
                    if assignment["status"] in {"completed", "failed"}
                    and not assignment.get("source_handoff_id")
                    and not self.assignment_has_handoffs(assignment["id"])
                ),
                None,
            )
            if disposable is None:
                raise ValueError("assignment_capacity_reached")
            self.assignments.remove(disposable)

    @_durable_mutation
    def set_running(self, assignment_id: str) -> dict[str, Any] | None:
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            return None
        now = _now()
        assignment.update(self._new_review())
        assignment["status"] = "running"
        assignment["attempt_id"] = uuid.uuid4().hex
        assignment["completed_run_id"] = ""
        assignment.pop("recovered_at", None)
        assignment["result"] = ""
        assignment["error"] = ""
        assignment["started_at"] = now
        assignment["completed_at"] = None
        assignment["updated_at"] = now
        self._sync_handoff_status(assignment, "running")
        return assignment

    @_durable_mutation
    def finish_assignment(
        self, assignment_id: str, *, result: str = "", error: str = ""
    ) -> dict[str, Any] | None:
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            return None
        if len(result) > MAX_RESULT_LENGTH:
            raise ValueError("assignment_result_too_long")
        now = _now()
        status = "failed" if error else "completed"
        assignment.update(self._new_review())
        assignment["status"] = status
        assignment["result"] = result
        assignment["error"] = error.strip()[:2_000]
        assignment["updated_at"] = now
        assignment["completed_at"] = now
        if assignment.get("attempt_id") and error != "run_output_persistence_failed":
            assignment["completed_run_id"] = assignment["attempt_id"]
        self._sync_handoff_status(assignment, status)
        return assignment

    @_durable_mutation
    def recover_output(self, assignment_id: str, run: dict) -> dict:
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            raise ValueError("assignment_not_found")
        attempt_id = assignment.get("attempt_id")
        if (
            not attempt_id
            or run.get("id") != attempt_id
            or run.get("assignment_id") != assignment_id
            or run.get("agent_id") != assignment["agent_id"]
            or run.get("trigger") != "assignment"
        ):
            raise ValueError("recovery_attempt_mismatch")
        if assignment.get("completed_run_id") == attempt_id:
            return assignment
        if assignment["status"] not in {"running", "failed"}:
            raise ValueError("assignment_not_recoverable")
        result, error = str(run.get("result") or ""), str(run.get("error") or "")
        if not error and not result.strip():
            error = "empty_agent_result"
        if len(result) > MAX_RESULT_LENGTH:
            result, error = "", "assignment_result_too_long"
        self.finish_assignment(assignment_id, result=result, error=error)
        assignment["completed_at"] = run["at"]
        assignment["completed_run_id"] = attempt_id
        assignment["recovered_at"] = _now()
        return assignment

    @staticmethod
    def _new_review() -> dict[str, Any]:
        return {
            "review_status": "unreviewed",
            "review_note": "",
            "reviewed_at": None,
            "review_version": uuid.uuid4().hex,
        }

    @_durable_mutation
    def review_assignment(
        self, assignment_id: str, *, decision: str, expected_version: str, note: str = ""
    ) -> dict[str, Any]:
        """Review the displayed artifact; never execute work or approve a Git write.

        The version changes on each execution and review decision, protecting
        against stale views in this single-process registry. It is not a lock
        across multiple server processes or authenticated reviewer attribution.
        """
        assignment = self.get_assignment(assignment_id)
        if not assignment:
            raise ValueError("assignment_not_found")
        if decision not in {"unreviewed", "accepted", "rejected"}:
            raise ValueError("invalid_review_decision")
        if len(note) > 2_000:
            raise ValueError("review_note_too_long")
        if assignment["status"] != "completed" or not assignment["result"].strip():
            raise ValueError("assignment_not_reviewable")
        if not expected_version or expected_version != assignment.get("review_version"):
            raise ValueError("review_changed_refresh_required")
        now = _now()
        assignment.update(
            review_status=decision,
            review_note=note.strip(),
            reviewed_at=now if decision != "unreviewed" else None,
            review_version=uuid.uuid4().hex,
            updated_at=now,
        )
        return assignment

    @_durable_mutation
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
        if len(source["result"]) > MAX_RESULT_LENGTH:
            raise ValueError("assignment_result_too_long")
        if len(self.handoffs) >= MAX_HANDOFFS:
            raise ValueError("handoff_capacity_reached")

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
            "artifact": source["result"],
            "instructions": instructions.strip()[:8_000],
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
        self.handoffs.insert(0, handoff)

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

    @_durable_mutation
    def delete_assignment(self, assignment_id: str) -> bool:
        assignment = self.get_assignment(assignment_id)
        if not assignment or assignment["status"] == "running":
            return False
        if self.assignment_has_handoffs(assignment_id):
            return False
        self.assignments.remove(assignment)
        return True

    def assignment_has_handoffs(self, assignment_id: str) -> bool:
        return any(
            handoff["source_assignment_id"] == assignment_id
            or handoff["target_assignment_id"] == assignment_id
            for handoff in self.handoffs
        )

    @_durable_mutation
    def delete_handoff(self, handoff_id: str) -> bool:
        handoff = self.get_handoff(handoff_id)
        if not handoff or handoff["status"] == "running":
            return False
        target = self.get_assignment(handoff["target_assignment_id"])
        if target:
            target["source_handoff_id"] = ""
            target["updated_at"] = _now()
        self.handoffs.remove(handoff)
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
        atomic_write_json(
            self.data_file,
            {"assignments": self.assignments, "handoffs": self.handoffs},
        )

    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            payload = json.loads(self.data_file.read_text())
            self.assignments[:] = payload.get("assignments", [])
            self.handoffs[:] = payload.get("handoffs", [])
            for assignment in self.assignments:
                # Deterministic for legacy files until the first new write.
                legacy_version = hashlib.sha256(
                    json.dumps(
                        [
                            assignment["id"],
                            assignment.get("completed_at"),
                            assignment.get("result", ""),
                        ]
                    ).encode()
                ).hexdigest()
                assignment.setdefault("review_status", "unreviewed")
                assignment.setdefault("review_note", "")
                assignment.setdefault("reviewed_at", None)
                assignment.setdefault("review_version", legacy_version)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load assignments: %s", exc)
            return
        self._recover_interrupted()

    @_durable_mutation
    def _recover_interrupted(self) -> None:
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
