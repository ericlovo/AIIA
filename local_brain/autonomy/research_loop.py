"""
Research Scheduler — drives the autonomous research loop on a cadence.

Every research cycle (typically nightly, via scripts/research_runner.py):
1. Load all active research topics from the TopicStore
2. Pick the least-recently-worked ones (so attention rotates fairly)
3. Run N sessions on each, up to a per-cycle topic budget
4. Skip paused/complete topics; never raise out of the loop — one bad
   topic must not stop the others

The scheduler owns *scheduling* only. Each session is the existing
ResearchEngine.run(topic) generator; we drain it to completion and let it
persist synthesis, gaps, and run bookkeeping as it already does. That keeps
this loop a thin, replay-safe orchestrator over machinery that already works
interactively via POST /v1/research/topics/{id}/run.

Gated behind AutonomyConfig: ships disabled (level != "phase2" or
research_enabled False → no-op).
"""

import logging
import time
from typing import Any

from local_brain.config import AutonomyConfig

logger = logging.getLogger("aiia.autonomy.research_loop")

# Topics whose last_run is None (never worked) sort first.
_NEVER_RUN = ""


class ResearchScheduler:
    """
    Autonomous research cadence over a TopicStore + ResearchEngine.

    Construction takes the engine and store explicitly (not a global) so the
    runner wires whatever data dir / model is configured, and tests can pass
    fakes with no Ollama or ChromaDB.
    """

    def __init__(
        self,
        config: AutonomyConfig,
        engine: Any,
        topic_store: Any,
        notify_fn: Any = None,
    ):
        self.config = config
        self.engine = engine
        self.topic_store = topic_store
        self._notify = notify_fn

    @property
    def enabled(self) -> bool:
        return self.config.level == "phase2" and self.config.research_enabled

    def _due_topics(self) -> list:
        """Active topics, least-recently-worked first."""
        active = [t for t in self.topic_store.list_all() if t.status == "active"]
        # last_run is an ISO-8601 string or None; None (never run) sorts first
        # because "" < any real timestamp.
        active.sort(key=lambda t: t.last_run or _NEVER_RUN)
        return active

    async def run_cycle(self) -> dict[str, Any]:
        """
        Run one research cycle: N sessions across the most-stale active topics.

        Returns a summary dict. Never raises — per-topic failures are counted
        and logged so the cycle (and any scheduler driving it) keeps going.
        """
        if not self.enabled:
            return {"skipped": True, "reason": "research_disabled"}

        results: dict[str, Any] = {
            "topics_worked": 0,
            "sessions_run": 0,
            "errors": 0,
            "topics": [],
        }

        due = self._due_topics()
        if not due:
            logger.info("Research cycle: no active topics to work")
            return results

        budget = max(0, self.config.research_max_topics_per_cycle)
        sessions_each = max(1, self.config.research_sessions_per_topic)

        for topic in due[:budget]:
            topic_summary = {
                "id": topic.id,
                "title": topic.title,
                "sessions": 0,
                "errors": 0,
            }
            for _ in range(sessions_each):
                try:
                    await self._run_one_session(topic)
                    topic_summary["sessions"] += 1
                    results["sessions_run"] += 1
                except Exception as e:
                    logger.warning(f"Research session failed for topic {topic.id}: {e}")
                    topic_summary["errors"] += 1
                    results["errors"] += 1
                    break  # stop hammering a topic that just failed

            results["topics_worked"] += 1
            results["topics"].append(topic_summary)

        if self._notify:
            await self._notify("research_cycle", results)

        logger.info(
            "Research cycle: worked %d topic(s), ran %d session(s), %d error(s)",
            results["topics_worked"],
            results["sessions_run"],
            results["errors"],
        )
        return results

    async def _run_one_session(self, topic: Any) -> dict[str, Any]:
        """
        Drain one ResearchEngine.run(topic) session to completion.

        Returns the terminal 'done' event if the engine emitted one. The
        engine persists synthesis/gaps/run_count itself, so we only need to
        consume the stream and surface a terminal/error signal.
        """
        start = time.monotonic()
        terminal: dict[str, Any] = {}
        async for event in self.engine.run(topic):
            kind = event.get("type")
            if kind == "done":
                terminal = event
            elif kind == "error":
                raise RuntimeError(event.get("message", "research session error"))
        elapsed = time.monotonic() - start
        logger.debug(
            "Session on %s finished in %.1fs (%s iterations)",
            topic.id,
            elapsed,
            terminal.get("iterations", "?"),
        )
        return terminal
