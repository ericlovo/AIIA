"""
GATED Downgrade Policy — Auto-promotes stale low-severity actions.

Promotes GATED actions to SUPERVISED after a configurable number of hours
(default 48h) without human action. Only applies to actions classified as
"low" severity by the upstream scanner.

Medium/high/critical severity actions remain GATED forever. Certain action
types (deploy_production, delete_data, modify_secrets) are never
auto-downgraded regardless of severity.

Typically runs every 6 hours via a scheduler.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from local_brain.config import AutonomyConfig

logger = logging.getLogger("aiia.autonomy.gated_downgrade")

# Actions that should NEVER be auto-downgraded, regardless of severity
NEVER_DOWNGRADE_TYPES = frozenset(
    {
        "deploy_production",
        "delete_data",
        "modify_secrets",
        "review",
    }
)


class GatedDowngradePolicy:
    """
    Auto-promote stale low-severity GATED actions to SUPERVISED.

    Depends on the command_center ActionQueue for action management and
    the SafetyGate tier definitions. The promotion mechanism writes a
    `tier_override` field on the live action so SafetyGate.get_tier()
    returns SUPERVISED instead of GATED — approve() alone is a no-op
    because the execution loop skips approved GATED actions.
    """

    def __init__(
        self,
        config: AutonomyConfig,
        action_queue: Any,
        notify_fn: Any = None,
    ):
        self.config = config
        self.action_queue = action_queue
        self._notify = notify_fn

    @property
    def enabled(self) -> bool:
        return self.config.level == "phase2" and self.config.gated_downgrade_enabled

    async def check_stale_gated_actions(self) -> Dict[str, Any]:
        """
        Scan for GATED actions older than downgrade_hours that haven't
        been acted on, and promote eligible ones to SUPERVISED.
        """
        if not self.enabled:
            return {"skipped": True, "reason": "gated_downgrade_disabled"}

        cutoff = datetime.utcnow() - timedelta(hours=self.config.gated_downgrade_hours)

        all_pending = self.action_queue.list_actions(status="pending", limit=100)

        promoted = 0
        skipped = 0

        for action in all_pending:
            created_str = action.get("created_at", "")
            if not created_str:
                continue

            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created.replace(tzinfo=None) > cutoff:
                    continue  # Not stale yet
            except (ValueError, TypeError):
                continue

            severity = action.get("severity", "error")
            if severity != self.config.gated_downgrade_max_severity:
                skipped += 1
                continue

            action_type = action.get("type", "")
            if action_type in NEVER_DOWNGRADE_TYPES:
                skipped += 1
                continue

            live_action = self.action_queue.get_action(action["id"])
            if not live_action:
                continue
            live_action["tier_override"] = "supervised"
            live_action["tier_override_reason"] = (
                f"auto_downgrade:stale_{self.config.gated_downgrade_hours}h"
            )
            self.action_queue.save()

            result = self.action_queue.approve(action["id"])
            if result:
                promoted += 1
                age_hours = (
                    datetime.utcnow() - created.replace(tzinfo=None)
                ).total_seconds() / 3600

                logger.info(
                    f"Auto-promoted GATED->SUPERVISED: {action.get('title', '')[:60]} "
                    f"(severity={severity}, age={age_hours:.0f}h)"
                )

                if self._notify:
                    await self._notify(
                        "gated_downgrade",
                        {
                            "action_id": action["id"],
                            "title": action.get("title", ""),
                            "severity": severity,
                            "age_hours": round(age_hours),
                        },
                    )

        return {
            "checked": len(all_pending),
            "promoted": promoted,
            "skipped": skipped,
            "cutoff_hours": self.config.gated_downgrade_hours,
        }
