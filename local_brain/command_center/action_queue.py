"""
Action Queue — Bridges AIIA's automated detection with Claude Code's engineering.

AIIA's scheduled tasks (code_health, test_runner, security_scan, learning_loop)
detect issues and create action items. Claude Code surfaces these at session start
via MCP tools. Nothing executes without Eric's explicit approval.

Lifecycle: pending -> approved -> executing -> completed
                                           -> failed
                   -> rejected
                   -> expired (auto after 72h)

Persistence: action_data.json alongside task_data.json.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.actions")

ACTION_DATA_FILE = Path(__file__).parent / "action_data.json"
MAX_ACTIONS = 200  # Cap total stored actions


class ActionQueue:
    """Manages actionable items with approval workflow and JSON persistence."""

    VALID_TYPES = {
        "lint_fix",
        "test_fix",
        "security_fix",
        "ci_fix",
        "review",
        "tech_debt",
        "post_commit_review",
        "verify_lint",
        "verify_test",
        "verify_security",
        "commit",
    }
    VALID_SEVERITIES = {"info", "warn", "error", "critical"}
    VALID_STATUSES = {
        "pending",
        "approved",
        "executing",
        "rejected",
        "completed",
        "failed",
        "expired",
    }

    def __init__(self):
        self.actions: List[Dict[str, Any]] = []
        self.load()

    def create_action(
        self,
        action_type: str,
        severity: str,
        title: str,
        description: str = "",
        proposed_fix: str = "",
        source_task: str = "",
        files_affected: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new pending action item."""
        action = {
            "id": uuid.uuid4().hex[:12],
            "type": action_type,
            "severity": severity,
            "title": title,
            "description": description,
            "proposed_fix": proposed_fix,
            "source_task": source_task,
            "status": "pending",
            "files_affected": files_affected or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "rejected_reason": None,
            "completed_result": None,
            "parent_id": None,
            "chain_on_complete": None,
            "execution_log_id": None,
            "execution_started_at": None,
            "retry_count": 0,
            "files_changed": [],
        }
        self.actions.insert(0, action)

        # Cap total actions
        if len(self.actions) > MAX_ACTIONS:
            self.actions = self.actions[:MAX_ACTIONS]

        self.save()
        logger.info(
            f"Action created: [{severity.upper()}] {action_type} — {title} (id={action['id']})"
        )
        return action

    def list_actions(
        self,
        status: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List actions, optionally filtered by status and/or type."""
        results = self.actions
        if status:
            results = [a for a in results if a["status"] == status]
        if action_type:
            results = [a for a in results if a["type"] == action_type]
        return results[:limit]

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get a single action by ID."""
        for action in self.actions:
            if action["id"] == action_id:
                return action
        return None

    def approve(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Mark an action as approved for execution."""
        action = self.get_action(action_id)
        if not action:
            return None
        if action["status"] != "pending":
            return action  # Already processed
        action["status"] = "approved"
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(f"Action approved: {action_id} — {action['title']}")
        return action

    def reject(self, action_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
        """Reject an action (won't fix)."""
        action = self.get_action(action_id)
        if not action:
            return None
        action["status"] = "rejected"
        action["rejected_reason"] = reason
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(f"Action rejected: {action_id} — {reason or 'no reason'}")
        return action

    def complete(self, action_id: str, result: str = "") -> Optional[Dict[str, Any]]:
        """Mark an approved/executing action as completed.

        If chain_on_complete is set, auto-creates a follow-up action.
        """
        action = self.get_action(action_id)
        if not action:
            return None
        action["status"] = "completed"
        action["completed_result"] = result
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(f"Action completed: {action_id}")

        # Chain: auto-create follow-up action if configured
        chain_type = action.get("chain_on_complete")
        if chain_type:
            chained = self.create_action(
                action_type=chain_type,
                severity=action["severity"],
                title=f"Chained from {action['title']}",
                description=(
                    f"Auto-created after completion of {action_id}"
                ),
                files_affected=action.get("files_affected", []),
            )
            chained["parent_id"] = action_id
            self.save()
            logger.info(
                f"Chain created: {chained['id']} ({chain_type})"
                f" from parent {action_id}"
            )

        return action

    def set_executing(
        self, action_id: str, execution_log_id: str
    ) -> Optional[Dict[str, Any]]:
        """Transition an approved action to executing state."""
        action = self.get_action(action_id)
        if not action or action["status"] != "approved":
            return None
        action["status"] = "executing"
        action["execution_log_id"] = execution_log_id
        action["execution_started_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(
            f"Action executing: {action_id} (log={execution_log_id})"
        )
        return action

    def get_approved(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return approved actions sorted by created_at (oldest first)."""
        approved = [
            a for a in self.actions if a["status"] == "approved"
        ]
        approved.sort(key=lambda a: a.get("created_at", ""))
        return approved[:limit]

    def fail_action(
        self, action_id: str, error: str
    ) -> Optional[Dict[str, Any]]:
        """Mark an action as failed with an error message."""
        action = self.get_action(action_id)
        if not action:
            return None
        action["status"] = "failed"
        action["completed_result"] = error
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(f"Action failed: {action_id} — {error}")
        return action

    def increment_retry(
        self, action_id: str
    ) -> Optional[Dict[str, Any]]:
        """Increment retry_count and reset status to approved."""
        action = self.get_action(action_id)
        if not action:
            return None
        action["retry_count"] = action.get("retry_count", 0) + 1
        action["status"] = "approved"
        action["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        logger.info(
            f"Action retry #{action['retry_count']}: {action_id}"
        )
        return action

    def get_chain_children(
        self, parent_id: str
    ) -> List[Dict[str, Any]]:
        """Return actions whose parent_id matches."""
        return [
            a
            for a in self.actions
            if a.get("parent_id") == parent_id
        ]

    def expire_old(self, hours: int = 72) -> int:
        """Expire pending actions older than `hours`. Returns count expired."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        expired_count = 0
        for action in self.actions:
            if action["status"] != "pending":
                continue
            created = datetime.fromisoformat(action["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                action["status"] = "expired"
                action["updated_at"] = datetime.now(timezone.utc).isoformat()
                expired_count += 1
        if expired_count:
            self.save()
            logger.info(f"Expired {expired_count} stale actions (>{hours}h old)")
        return expired_count

    def summary(self) -> Dict[str, Any]:
        """Count actions by status and severity."""
        by_status: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for action in self.actions:
            status = action["status"]
            severity = action["severity"]
            by_status[status] = by_status.get(status, 0) + 1
            if status == "pending":
                by_severity[severity] = by_severity.get(severity, 0) + 1
        return {
            "total": len(self.actions),
            "by_status": by_status,
            "pending_by_severity": by_severity,
        }

    # ─── Persistence ─────────────────────────────────────────

    def save(self):
        """Persist actions to JSON."""
        try:
            data = {
                "actions": self.actions,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            ACTION_DATA_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to persist action data: {e}")

    def load(self):
        """Load persisted actions from JSON."""
        if not ACTION_DATA_FILE.exists():
            logger.info("No action data file — starting with empty queue")
            return

        try:
            data = json.loads(ACTION_DATA_FILE.read_text())
            self.actions = data.get("actions", [])
            logger.info(f"Loaded {len(self.actions)} actions from disk")
        except Exception as e:
            logger.warning(f"Could not load action data: {e}")
            self.actions = []
