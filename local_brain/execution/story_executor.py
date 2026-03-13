"""
Story Executor — Decomposes kanban stories into executable actions.

Takes an "active" story from the roadmap, uses local LLM to break it into
actionable steps, creates actions in the queue, and tracks progress.

Story lifecycle:
  active → in_progress (decomposed, actions created)
  in_progress → shipped (all actions completed)
  in_progress → blocked (any action failed permanently)
"""

from __future__ import annotations

import glob as globmod
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from local_brain.ollama_client import OllamaClient
from local_brain.config import get_config

if TYPE_CHECKING:
    from local_brain.command_center.action_queue import (
        ActionQueue,
    )
    from local_brain.scripts.roadmap_store import (
        RoadmapStore,
    )

logger = logging.getLogger("aiia.execution.story")

DECOMPOSE_SYSTEM = """\
You are a technical project planner for a Python/FastAPI/React monorepo.

Given a story title and description, decompose it into concrete, executable action steps.

Each step must be one of these action types:
- lint_fix: Auto-fixable linting issues (ruff)
- test_fix: Fix failing tests
- security_fix: Fix security vulnerabilities
- ci_fix: Fix CI/CD issues
- tech_debt: Refactoring or cleanup
- review: Code review needed (human)

For each step, specify:
- type: one of the action types above
- severity: info | warn | error | critical
- title: short description of what to do
- description: detailed instructions
- proposed_fix: suggested approach or code changes
- files_affected: list of file paths that need changes

Respond ONLY with a JSON array of steps. No markdown, no explanation.
Example:
[
  {
    "type": "tech_debt",
    "severity": "warn",
    "title": "Extract config into dataclass",
    "description": "Move hardcoded values from main.py into a Config dataclass",
    "proposed_fix": "Create config.py with @dataclass Config class, import in main.py",
    "files_affected": ["local_brain/config.py", "local_brain/main.py"]
  }
]"""

DECOMPOSE_PROMPT = """\
Decompose this story into executable action steps:

Title: {title}
Product: {product}
Priority: {priority}
Description:
{description}

Break this into 2-8 concrete steps that can be executed independently or sequentially.
Focus on what code changes are needed. Be specific about file paths in the repo.
Respond with ONLY a JSON array."""


class StoryExecutor:
    def __init__(
        self,
        action_queue: ActionQueue,
        roadmap_store: RoadmapStore,
        ollama: OllamaClient | None = None,
    ):
        self._queue = action_queue
        self._roadmap = roadmap_store
        self._ollama = ollama or OllamaClient()
        self._config = get_config()

    async def execute_story(
        self, story_id: str, auto_approve: bool = False
    ) -> dict[str, Any]:
        """Decompose a story into actions and start execution.

        Args:
            story_id: Roadmap story ID
            auto_approve: If True, auto-approve all created actions

        Returns:
            Dict with status, action_ids, and step count
        """
        story = self._roadmap.get(story_id)
        if not story:
            return {"status": "error", "reason": "Story not found"}

        if story["status"] not in ("active", "backlog"):
            return {
                "status": "error",
                "reason": (
                    f"Story status is '{story['status']}', "
                    "expected 'active' or 'backlog'"
                ),
            }

        # Decompose via local LLM
        steps = await self._decompose(story)
        if not steps:
            return {
                "status": "error",
                "reason": "LLM decomposition returned no steps",
            }

        # Create actions for each step
        action_ids = []
        repo_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        for i, step in enumerate(steps):
            # Expand glob patterns in files_affected to real paths
            raw_files = step.get("files_affected", [])
            files = self._expand_file_globs(raw_files, repo_path)

            action = self._queue.create_action(
                action_type=step.get("type", "tech_debt"),
                severity=step.get("severity", "info"),
                title=step.get("title", f"Step {i + 1}"),
                description=step.get("description", ""),
                proposed_fix=step.get("proposed_fix", ""),
                source_task=f"story:{story_id}",
                files_affected=files,
            )
            action["story_id"] = story_id
            action["story_step"] = i + 1
            action["story_step_total"] = len(steps)
            self._queue.save()

            if auto_approve:
                self._queue.approve(action["id"])

            action_ids.append(action["id"])

        # Transition story to in_progress
        self._roadmap.update(story_id, status="in_progress")

        logger.info(
            f"Story {story_id} decomposed into {len(steps)} actions: {action_ids}"
        )

        return {
            "status": "decomposed",
            "story_id": story_id,
            "steps": len(steps),
            "action_ids": action_ids,
            "auto_approved": auto_approve,
        }

    async def _decompose(self, story: dict) -> list[dict]:
        """Use local LLM to decompose a story into action steps."""
        prompt = DECOMPOSE_PROMPT.format(
            title=story.get("title", ""),
            product=story.get("product", ""),
            priority=story.get("priority", "P2"),
            description=story.get("description", "No description"),
        )

        model_config = self._config.models.get("task")
        model = model_config.model_name if model_config else "llama3.1:8b-instruct-q8_0"

        try:
            response = await self._ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                system=DECOMPOSE_SYSTEM,
                temperature=0.3,  # Low for structured output
                max_tokens=2048,
                num_ctx=8192,
            )
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return []

        content = response.get("message", {}).get("content", "")
        return self._parse_steps(content)

    def _parse_steps(self, content: str) -> list[dict]:
        """Parse LLM output into list of action step dicts."""
        content = content.strip()

        # Try to extract JSON array from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    content = part
                    break

        try:
            steps = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON array in the content
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end > start:
                try:
                    steps = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse decomposition output: {content[:200]}"
                    )
                    return []
            else:
                logger.warning(
                    f"No JSON array found in decomposition output: {content[:200]}"
                )
                return []

        if not isinstance(steps, list):
            return []

        # Validate each step
        valid_types = {
            "lint_fix",
            "test_fix",
            "security_fix",
            "ci_fix",
            "tech_debt",
            "review",
        }
        validated = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("type") not in valid_types:
                step["type"] = "tech_debt"
            if step.get("severity") not in ("info", "warn", "error", "critical"):
                step["severity"] = "info"
            if not step.get("title"):
                continue
            validated.append(step)

        return validated

    def get_story_progress(self, story_id: str) -> dict[str, Any]:
        """Get execution progress for a story."""
        story = self._roadmap.get(story_id)
        if not story:
            return {"status": "error", "reason": "Story not found"}

        actions = self._get_story_actions(story_id)
        if not actions:
            return {
                "story": story,
                "actions": [],
                "progress": 0,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "pending": 0,
            }

        total = len(actions)
        completed = sum(1 for a in actions if a["status"] == "completed")
        failed = sum(1 for a in actions if a["status"] == "failed")
        pending = sum(
            1 for a in actions if a["status"] in ("pending", "approved", "executing")
        )

        progress = int((completed / total) * 100) if total > 0 else 0

        return {
            "story": story,
            "actions": [
                {
                    "id": a["id"],
                    "type": a.get("type", ""),
                    "title": a.get("title", ""),
                    "status": a["status"],
                    "step": a.get("story_step", 0),
                }
                for a in actions
            ],
            "progress": progress,
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
        }

    def check_story_completion(self, story_id: str) -> str | None:
        """Check if a story should transition based on action states.

        Returns:
            New status ("shipped" or "blocked") or None if still in progress.
        """
        actions = self._get_story_actions(story_id)
        if not actions:
            return None

        statuses = [a["status"] for a in actions]

        # All completed → ship it
        if all(s == "completed" for s in statuses):
            self._roadmap.update(story_id, status="shipped")
            logger.info(f"Story {story_id} shipped — all actions completed")
            return "shipped"

        # Any permanently failed → block
        failed = [
            a
            for a in actions
            if a["status"] == "failed"
            and a.get("retry_count", 0) >= self._config.execution_max_retries
        ]
        if failed:
            titles = [a.get("title", a["id"]) for a in failed]
            self._roadmap.update(story_id, status="blocked")
            logger.warning(
                f"Story {story_id} blocked — failed actions: {', '.join(titles)}"
            )
            return "blocked"

        return None

    def _get_story_actions(self, story_id: str) -> list[dict]:
        """Get all actions belonging to a story."""
        return [
            a
            for a in self._queue.actions
            if a.get("story_id") == story_id
            or a.get("source_task") == f"story:{story_id}"
        ]

    @staticmethod
    def _expand_file_globs(files: list[str], repo_path: str) -> list[str]:
        """Expand glob patterns (*.py) to real file paths, relative to repo."""
        expanded = []
        for f in files:
            if "*" in f or "?" in f:
                full = os.path.join(repo_path, f)
                matches = globmod.glob(full)
                for m in matches:
                    if os.path.isfile(m):
                        rel = os.path.relpath(m, repo_path)
                        expanded.append(rel)
            else:
                expanded.append(f)
        return expanded
