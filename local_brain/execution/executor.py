from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from .chains import VERIFY_TYPES, apply_chain
from .execution_log import ExecutionLog
from .git_ops import GitOps
from .safety import SafetyGate, SafetyTier
from .story_executor import StoryExecutor
from .strategies import (
    ClaudeCodeStrategy,
    CommitStrategy,
    DirectFixStrategy,
    select_strategy,
)
from .subprocess_pool import ExecutionTimeout, SubprocessPool
from .verification import Verifier

logger = logging.getLogger("aiia.execution.engine")

REPO_PATH = os.getenv("AIIA_REPO_PATH", "/path/to/AIIA")


class ExecutionEngine:
    def __init__(
        self,
        action_queue,  # ActionQueue instance
        config,  # LocalBrainConfig instance
        broadcast_fn: Callable | None = None,
        roadmap_store=None,  # RoadmapStore instance (optional)
    ):
        self._queue = action_queue
        self._config = config
        self._broadcast = broadcast_fn
        self._task: asyncio.Task | None = None

        repo_path = REPO_PATH
        self.safety = SafetyGate()
        self.pool = SubprocessPool(max_concurrent=1)
        self.execution_log = ExecutionLog(data_dir=config.execution_data_dir)
        self._git = GitOps(repo_path, self.pool)
        self._verifier = Verifier(self.pool, repo_path)
        self._direct = DirectFixStrategy(self.pool, repo_path, self._verifier)
        self._claude = ClaudeCodeStrategy(
            self.pool,
            repo_path,
            self._verifier,
            config.execution_branch_prefix,
            git_ops=self._git,
        )
        self._commit = CommitStrategy(self._git)

        # Story execution
        self._roadmap = roadmap_store
        self._story_executor: StoryExecutor | None = None
        if roadmap_store:
            self._story_executor = StoryExecutor(action_queue, roadmap_store)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def kill_switch(self) -> bool:
        return self.safety.kill_switch

    @kill_switch.setter
    def kill_switch(self, value: bool) -> None:
        self.safety.kill_switch = value

    async def start(self) -> None:
        if self.is_running:
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Execution engine started")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.pool.kill_all()
        logger.info("Execution engine stopped")

    async def _run_loop(self) -> None:
        poll = self._config.execution_poll_interval
        logger.info(f"Execution loop running (poll every {poll}s)")
        while True:
            try:
                if not self._config.execution_enabled or self.safety.kill_switch:
                    await asyncio.sleep(poll)
                    continue

                approved = self._queue.get_approved(limit=10)
                if not approved:
                    await asyncio.sleep(poll)
                    continue

                # Skip GATED actions in the auto-loop
                action = None
                for candidate in approved:
                    tier = self.safety.get_tier(candidate)
                    if tier != SafetyTier.GATED:
                        action = candidate
                        break

                if not action:
                    await asyncio.sleep(poll)
                    continue

                await self.execute_action(action)
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Execution loop error: {e}", exc_info=True)
                await asyncio.sleep(poll)

    async def execute_action(self, action: dict) -> dict[str, Any]:
        action_id = action["id"]
        action_type = action.get("type", "unknown")

        # Safety check
        can, reason = self.safety.can_execute(action)
        if not can:
            logger.warning(f"Safety blocked {action_id}: {reason}")
            self._queue.fail_action(action_id, f"Safety: {reason}")
            return {"status": "blocked", "reason": reason}

        tier = self.safety.get_tier(action)

        # GATED actions need explicit execute_now()
        if tier == SafetyTier.GATED:
            logger.info(f"Skipping GATED action {action_id} — needs explicit trigger")
            return {"status": "gated", "action_id": action_id}

        # SUPERVISED: 30s notification window
        if tier == SafetyTier.SUPERVISED:
            self._emit(
                "execution_pending",
                action_id=action_id,
                action_type=action_type,
                countdown=30,
                message=(
                    f"SUPERVISED: {action.get('title', '')} "
                    f"— executing in 30s, kill switch to abort"
                ),
            )
            await asyncio.sleep(30)
            if self.safety.kill_switch:
                return {
                    "status": "aborted",
                    "action_id": action_id,
                    "reason": "Kill switch during SUPERVISED countdown",
                }

        return await self._do_execute(action, tier.value)

    async def execute_now(self, action_id: str) -> dict[str, Any]:
        action = self._queue.get_action(action_id)
        if not action:
            return {"status": "error", "reason": "Action not found"}
        if action["status"] not in ("approved", "executing"):
            return {
                "status": "error",
                "reason": (
                    f"Action status is '{action['status']}', expected 'approved'"
                ),
            }

        can, reason = self.safety.can_execute(action)
        if not can:
            return {"status": "blocked", "reason": reason}

        return await self._do_execute(action, "gated_explicit")

    async def _do_execute(self, action: dict, tier_label: str) -> dict[str, Any]:
        action_id = action["id"]
        action_type = action.get("type", "unknown")

        # Handle verify_* actions — just run verification, no fix
        if action_type in VERIFY_TYPES:
            return await self._handle_verify_only(action, tier_label)

        strategy = select_strategy(action, self._direct, self._claude, self._commit)
        strategy_name = "direct"
        if strategy is self._claude:
            strategy_name = "claude_code"
        elif strategy is self._commit:
            strategy_name = "commit"

        # Start log record
        log_record = self.execution_log.start(
            action_id=action_id,
            action_type=action_type,
            strategy=strategy_name,
            safety_tier=tier_label,
            input_summary=action.get("title", ""),
        )

        # Mark executing
        self._queue.set_executing(action_id, log_record.id)
        self._emit(
            "execution_started",
            action_id=action_id,
            action_type=action_type,
            strategy=strategy_name,
        )

        try:
            result = await asyncio.wait_for(
                strategy.execute(action),
                timeout=self._config.execution_max_timeout,
            )
        except ExecutionTimeout as e:
            self._handle_failure(
                action, log_record.id, f"Timeout: {e}", is_timeout=True
            )
            return {"status": "timeout", "action_id": action_id}
        except Exception as e:
            self._handle_failure(action, log_record.id, str(e))
            return {
                "status": "failed",
                "action_id": action_id,
                "error": str(e),
            }

        # Verify if strategy supports it
        verified = None
        if strategy.has_verification and result.success:
            try:
                verified = await strategy.verify(action, result)
            except Exception as e:
                logger.warning(f"Verification error for {action_id}: {e}")
                verified = None

        if result.success and verified is not False:
            self.execution_log.complete(
                log_record.id,
                status="success",
                output_summary=result.output,
                files_changed=result.files_changed,
                verified=verified,
            )
            self._queue.complete(action_id, result=result.output[:500])

            # Check story completion
            self._check_story_transition(action)

            # Auto-commit if configured
            commit_hash = None
            if self._config.execution_auto_commit and result.files_changed:
                try:
                    await self._git.stage_files(result.files_changed)
                    commit_hash = await self._git.commit(
                        f"fix({action_type}): {action.get('title', 'automated fix')}"
                    )
                except Exception as e:
                    logger.warning(f"Auto-commit failed for {action_id}: {e}")

            # Chain: create follow-up action if defined
            chained = apply_chain(self._queue, action)
            if chained:
                self._emit(
                    "chain_created",
                    parent_id=action_id,
                    chained_id=chained["id"],
                    chained_type=chained.get("type", ""),
                )

            self._emit(
                "execution_complete",
                action_id=action_id,
                success=True,
                verified=verified,
                commit_hash=commit_hash,
            )
            return {
                "status": "success",
                "action_id": action_id,
                "verified": verified,
                "files_changed": result.files_changed,
                "commit_hash": commit_hash,
            }
        else:
            error = result.error or "Fix did not pass verification"
            self._handle_failure(action, log_record.id, error)
            return {
                "status": "failed",
                "action_id": action_id,
                "error": error,
            }

    async def _handle_verify_only(
        self, action: dict, tier_label: str
    ) -> dict[str, Any]:
        """Handle verify_* actions — run verification check, no fix."""
        action_id = action["id"]
        action_type = action.get("type", "unknown")

        # Map verify_lint → lint_fix for the verifier
        real_type = action_type.replace("verify_", "") + "_fix"
        verify_action = {**action, "type": real_type}

        log_record = self.execution_log.start(
            action_id=action_id,
            action_type=action_type,
            strategy="verify_only",
            safety_tier=tier_label,
            input_summary=action.get("title", ""),
        )
        self._queue.set_executing(action_id, log_record.id)

        try:
            v_result = await self._verifier.verify(verify_action)
        except Exception as e:
            self._handle_failure(action, log_record.id, str(e))
            return {
                "status": "failed",
                "action_id": action_id,
                "error": str(e),
            }

        if v_result.verified is True:
            self.execution_log.complete(
                log_record.id,
                status="success",
                output_summary=v_result.output,
                verified=True,
            )
            self._queue.complete(action_id, result=v_result.reason)
            self._emit(
                "execution_complete",
                action_id=action_id,
                success=True,
                verified=True,
            )
            return {
                "status": "success",
                "action_id": action_id,
                "verified": True,
            }
        else:
            reason = v_result.reason or "Verification failed"
            self._handle_failure(action, log_record.id, reason)
            return {
                "status": "failed",
                "action_id": action_id,
                "error": reason,
            }

    def _handle_failure(
        self,
        action: dict,
        log_id: str,
        error: str,
        is_timeout: bool = False,
    ) -> None:
        action_id = action["id"]
        retry_count = action.get("retry_count", 0)
        max_retries = self._config.execution_max_retries

        self.execution_log.complete(
            log_id,
            status="timeout" if is_timeout else "failed",
            error=error,
        )

        if retry_count < max_retries:
            self._queue.increment_retry(action_id)
            logger.info(f"Retrying {action_id} ({retry_count + 1}/{max_retries})")
        else:
            self._queue.fail_action(action_id, error)
            logger.warning(f"Action {action_id} failed permanently: {error}")
            # Check if story should be blocked
            self._check_story_transition(action)

        self._emit(
            "execution_failed",
            action_id=action_id,
            error=error,
        )

    async def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self._config.execution_enabled,
            "is_running": self.is_running,
            "kill_switch": self.safety.kill_switch,
            "active_subprocesses": self.pool.active_count,
            "recent": self.execution_log.list_recent(5),
            "stats": self.execution_log.get_stats(),
        }

    async def emergency_stop(self) -> None:
        self.safety.kill_switch = True
        killed = await self.pool.kill_all()
        for action in self._queue.list_actions(status="executing"):
            self._queue.fail_action(
                action["id"],
                "Emergency stop — kill switch engaged",
            )
        logger.warning(f"Emergency stop: killed {killed} processes")

    async def execute_story(
        self, story_id: str, auto_approve: bool = False
    ) -> dict[str, Any]:
        """Decompose and execute a story from the kanban board."""
        if not self._story_executor:
            return {
                "status": "error",
                "reason": "Story executor not initialized (no roadmap store)",
            }
        return await self._story_executor.execute_story(
            story_id, auto_approve=auto_approve
        )

    def get_story_progress(self, story_id: str) -> dict[str, Any]:
        """Get execution progress for a story."""
        if not self._story_executor:
            return {"status": "error", "reason": "No story executor"}
        return self._story_executor.get_story_progress(story_id)

    def _check_story_transition(self, action: dict) -> None:
        """Check if a completed/failed action's parent story should transition."""
        if not self._story_executor:
            return
        story_id = action.get("story_id")
        if not story_id:
            # Check source_task for story: prefix
            source = action.get("source_task", "")
            if source.startswith("story:"):
                story_id = source[6:]
        if not story_id:
            return

        new_status = self._story_executor.check_story_completion(story_id)
        if new_status:
            self._emit(
                "story_transition",
                story_id=story_id,
                new_status=new_status,
            )

    def _emit(self, event_type: str, **data: Any) -> None:
        if not self._broadcast:
            return
        try:
            import json

            msg = json.dumps({"type": event_type, **data})
            result = self._broadcast(msg)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
