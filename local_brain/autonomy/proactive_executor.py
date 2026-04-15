"""
Proactive Story Executor — Auto-decomposes and queues stories for execution.

Extends the StoryExecutor with time-gated, confidence-gated proactive
execution for P0/P1 stories. Runs on a schedule (every 30 minutes via
TaskRunner in a typical deployment). Only executes outside business
hours, and only when an optional production health check passes.

All queued actions land at SUPERVISED tier (human notified, not blocked).
"""

import logging
from datetime import datetime, time
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from local_brain.config import AutonomyConfig

logger = logging.getLogger("aiia.autonomy.proactive")


class ProactiveStoryExecutor:
    """
    Auto-decomposes and queues P0/P1 stories for execution when:
    - AutonomyConfig.proactive_story_execution is True
    - Outside configured business hours
    - Optional production health check passes (skipped if URL not configured)
    - Story priority is in the allowed set
    """

    def __init__(
        self,
        config: AutonomyConfig,
        story_executor: Any,
        action_queue: Any,
        roadmap_store: Any,
        notify_fn: Any = None,
    ):
        self.config = config
        self.story_executor = story_executor
        self.action_queue = action_queue
        self.roadmap_store = roadmap_store
        self._notify = notify_fn

    @property
    def enabled(self) -> bool:
        return self.config.level == "phase2" and self.config.proactive_story_execution

    async def evaluate_pending_stories(self) -> Dict[str, Any]:
        """
        Main entry point. Returns a summary of what was evaluated and queued.
        """
        if not self.enabled:
            return {"skipped": True, "reason": "proactive_execution_disabled"}

        if self._is_business_hours():
            return {"skipped": True, "reason": "business_hours"}

        if not await self._production_healthy():
            return {"skipped": True, "reason": "production_unhealthy"}

        stories = self._get_eligible_stories()
        if not stories:
            return {"evaluated": 0, "queued": 0}

        queued = 0
        for story in stories[:2]:  # Max 2 concurrent stories
            try:
                result = await self.story_executor.execute_story(
                    story_id=story["id"],
                    auto_approve=True,  # SUPERVISED tier, not GATED
                )
                if result.get("status") == "decomposed":
                    queued += 1
                    if self._notify:
                        await self._notify(
                            "proactive_story_queued",
                            {
                                "story_id": story["id"],
                                "title": story.get("title", ""),
                                "actions": result.get("actions_created", 0),
                                "priority": story.get("priority", ""),
                            },
                        )
            except Exception as e:
                logger.warning(f"Failed to decompose story {story.get('id')}: {e}")

        return {
            "evaluated": len(stories),
            "queued": queued,
        }

    def _is_business_hours(self) -> bool:
        tz_name = self.config.proactive_business_hours_tz
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            logger.warning(f"Unknown timezone '{tz_name}', falling back to UTC")
            tz = ZoneInfo("UTC")
        now = datetime.now(tz).time()
        start = time(self.config.proactive_business_hours_start, 0)
        end = time(self.config.proactive_business_hours_end, 0)
        return start <= now <= end

    async def _production_healthy(self) -> bool:
        """
        Check the configured production health endpoint before running
        autonomous actions. Returns True if no URL is configured (the
        gate is optional — operators who don't want it can leave the
        env var unset and the executor will run whenever it's outside
        business hours).
        """
        url = self.config.proactive_health_check_url
        if not url:
            return True

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False  # Fail closed when the probe errors

    def _get_eligible_stories(self) -> List[Dict[str, Any]]:
        """Get stories matching allowed priorities that aren't already in-progress."""
        if not self.roadmap_store:
            return []

        all_stories = (
            self.roadmap_store.get_stories(statuses=["active", "backlog"])
            if hasattr(self.roadmap_store, "get_stories")
            else []
        )

        eligible = [
            s
            for s in all_stories
            if s.get("priority") in self.config.proactive_priorities
            and s.get("status") != "in_progress"
        ]

        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        eligible.sort(key=lambda s: priority_order.get(s.get("priority", "P3"), 3))

        return eligible
