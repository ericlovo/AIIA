"""
Self-Healing Production Monitor — Detects and alerts on unhealthy services.

Monitors a configured list of services. On unhealthy detection:
1. Increments a per-service consecutive-failure counter
2. After N consecutive failures, creates a tech_debt action in the queue
3. Searches AIIA memory for known fix patterns (best-effort)
4. Notifies via the optional broadcast function

Services to monitor are loaded from AIIA_SERVICES_CONFIG (a JSON file)
into AutonomyConfig.monitored_services. If no services are configured,
the monitor runs as a no-op. This keeps the public default safe — the
operator must opt in to monitor anything.

Typically runs every 5 minutes via a scheduler.
"""

import logging
from typing import Any, Dict, Optional

from local_brain.config import AutonomyConfig

logger = logging.getLogger("aiia.autonomy.self_healing")

# Known fix patterns: detection condition -> remediation metadata
KNOWN_FIX_PATTERNS = {
    "health_timeout": {
        "detection": lambda status: status.get("health_check_failures", 0) > 3,
        "action_type": "tech_debt",
        "severity": "error",
        "title_template": "[AUTO] Health timeout: {service_name}",
        "description_template": (
            "Service {service_name} has failed {failures} consecutive health checks. "
            "Automated restart recommended."
        ),
    },
    "high_error_rate": {
        "detection": lambda status: status.get("error_rate_5m", 0) > 0.1,
        "action_type": "tech_debt",
        "severity": "error",
        "title_template": "[AUTO] High error rate: {service_name} ({error_rate:.1%})",
        "description_template": (
            "Service {service_name} error rate is {error_rate:.1%} over last 5 minutes. "
            "Check logs for root cause."
        ),
    },
    "deploy_stale": {
        "detection": lambda status: status.get("deploy_age_hours", 0) > 168,  # 7 days
        "action_type": "tech_debt",
        "severity": "warn",
        "title_template": "[AUTO] Stale deploy: {service_name} ({age_days}d old)",
        "description_template": (
            "Service {service_name} last deployed {age_days} days ago. "
            "Consider redeploying to pick up dependency updates."
        ),
    },
}


class SelfHealingMonitor:
    """
    Production health monitor with automated action-queue integration.

    The list of services comes from config.monitored_services — populated
    from AIIA_SERVICES_CONFIG at startup. Each entry is a dict with at
    minimum `name` and `url` fields; the URL is probed via HTTP GET.
    """

    def __init__(
        self,
        config: AutonomyConfig,
        action_queue: Any,
        memory: Optional[Any] = None,
        notify_fn: Any = None,
    ):
        self.config = config
        self.action_queue = action_queue
        self.memory = memory
        self._notify = notify_fn
        self._consecutive_failures: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.config.level == "phase2" and self.config.self_healing_enabled

    async def check_and_heal(self) -> Dict[str, Any]:
        """Check every configured service and take action on unhealthy ones."""
        if not self.enabled:
            return {"skipped": True, "reason": "self_healing_disabled"}

        results = {
            "services_checked": 0,
            "healthy": 0,
            "unhealthy": 0,
            "actions_created": 0,
        }

        for service in self.config.monitored_services:
            if "name" not in service or "url" not in service:
                continue

            status = await self._check_service(service)
            results["services_checked"] += 1

            if status["healthy"]:
                results["healthy"] += 1
                self._consecutive_failures[service["name"]] = 0
            else:
                results["unhealthy"] += 1
                self._consecutive_failures[service["name"]] = (
                    self._consecutive_failures.get(service["name"], 0) + 1
                )

                # Only act after 2+ consecutive failures (avoid flapping)
                if self._consecutive_failures[service["name"]] >= 2:
                    actions = await self._handle_unhealthy(service, status)
                    results["actions_created"] += actions

        return results

    async def _check_service(self, service: Dict[str, str]) -> Dict[str, Any]:
        """Check a single service's health endpoint."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(service["url"])
                return {
                    "healthy": resp.status_code == 200,
                    "status_code": resp.status_code,
                    "latency_ms": resp.elapsed.total_seconds() * 1000 if resp.elapsed else 0,
                }
        except Exception as e:
            return {
                "healthy": False,
                "status_code": 0,
                "error": str(e),
                "latency_ms": 0,
            }

    async def _handle_unhealthy(
        self,
        service: Dict[str, str],
        status: Dict[str, Any],
    ) -> int:
        """Handle an unhealthy service — create action queue entries."""
        service_name = service["name"]
        failures = self._consecutive_failures.get(service_name, 0)
        actions_created = 0

        title = f"[AUTO] Service unhealthy: {service_name} ({failures} failures)"
        description = (
            f"Service {service_name} has been unhealthy for {failures} consecutive checks.\n"
            f"Status: {status.get('status_code', 'unknown')}\n"
            f"Error: {status.get('error', 'none')}\n"
            f"Latency: {status.get('latency_ms', 0):.0f}ms"
        )

        # ActionQueue uses action_type= (not type=) for filtering
        existing = self.action_queue.list_actions(status="pending", action_type="tech_debt")
        already_exists = any(service_name in a.get("title", "") for a in existing)

        if not already_exists:
            self.action_queue.create_action(
                action_type="tech_debt",
                severity="error",
                title=title,
                description=description,
                proposed_fix=f"Investigate {service_name} health endpoint and service logs",
                source_task="self_healing_monitor",
                files_affected=[],
            )
            actions_created += 1

        if self.memory and hasattr(self.memory, "search"):
            try:
                matches = self.memory.search(
                    f"fix unhealthy {service_name}",
                    categories=["patterns", "lessons"],
                    limit=3,
                )
                if matches:
                    logger.info(f"Found {len(matches)} memory matches for {service_name} fix")
            except Exception:
                pass  # Memory search is best-effort

        if self._notify:
            await self._notify(
                "service_unhealthy",
                {
                    "service": service_name,
                    "failures": failures,
                    "status": status,
                    "actions_created": actions_created,
                },
            )

        logger.warning(
            f"Service unhealthy: {service_name} "
            f"(failures={failures}, actions_created={actions_created})"
        )

        return actions_created
