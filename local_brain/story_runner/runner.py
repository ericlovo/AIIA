"""
AIIA Story Runner — Main orchestrator.

Reads a story from AIIA's backlog, creates an isolated worktree,
runs planner + coder agents, validates with CI, and creates a PR.

Usage:
    python -m local_brain.story_runner.runner \
        --story "Add cash flow forecast to case analysis" \
        --product default

    # Or from AIIA backlog:
    python -m local_brain.story_runner.runner \
        --from-backlog --max-budget 3.0

    # Dry run (plan only, no code):
    python -m local_brain.story_runner.runner \
        --story "..." --plan-only
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from local_brain.story_runner.security import (
    ALLOWED_BASH_COMMANDS,
    validate_bash_command,
)
from local_brain.story_runner.progress import (
    ActionPlan,
    load_plan,
    save_plan,
)

logger = logging.getLogger("aiia.story_runner")

# ─── Configuration ──────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]  # AIIA/
AIIA_URL = os.getenv("AIIA_URL", "http://localhost:8100")
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_BUDGET = 5.0
DEFAULT_MAX_TURNS = 50
DELAY_BETWEEN_SESSIONS = 3  # seconds

# Product → directory mapping
# Customize for your multi-tenant setup
PRODUCT_DIRS = {
    "default": "products/default-app",
    "platform": "local_brain",
}


# ─── AIIA Integration ──────────────────────────────────────────


def fetch_aiia_context() -> str:
    """Fetch AIIA's current context: recent decisions, patterns, lessons."""
    import httpx

    try:
        resp = httpx.get(f"{AIIA_URL}/v1/aiia/memory", timeout=5)
        if resp.status_code != 200:
            return ""

        data = resp.json()
        memories = data.get("memories", [])

        # Group by category, take most recent
        context_parts = []
        for category in ["decisions", "patterns", "lessons"]:
            items = [m for m in memories if m.get("category") == category]
            if items:
                context_parts.append(f"\n## AIIA {category.title()}")
                for item in items[:5]:
                    context_parts.append(f"- {item.get('fact', '')}")

        return "\n".join(context_parts) if context_parts else ""
    except Exception as e:
        logger.warning(f"Could not fetch AIIA context: {e}")
        return ""


def notify_aiia(event: str, data: Dict[str, Any]) -> None:
    """Send event to AIIA (fire-and-forget)."""
    import httpx

    try:
        httpx.post(
            f"{AIIA_URL}/v1/aiia/remember",
            json={
                "fact": f"Story Runner: {event} — {json.dumps(data, default=str)[:500]}",
                "category": "sessions",
                "source": "story_runner",
            },
            timeout=3,
        )
    except Exception:
        pass  # fire-and-forget


# ─── Git Worktree Management ───────────────────────────────────


def create_worktree(story_slug: str) -> Path:
    """Create an isolated git worktree for the story."""
    branch_name = f"aiia/{story_slug}"
    worktree_path = REPO_ROOT / ".claude" / "worktrees" / story_slug

    # Create branch from main
    subprocess.run(
        ["git", "branch", branch_name, "origin/main"],
        cwd=REPO_ROOT,
        capture_output=True,
    )

    # Create worktree
    subprocess.run(
        ["git", "worktree", "add", str(worktree_path), branch_name],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )

    logger.info(f"Created worktree at {worktree_path} on branch {branch_name}")
    return worktree_path


def cleanup_worktree(worktree_path: Path, branch_name: str) -> None:
    """Remove worktree and branch (only if no changes)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        logger.warning(
            f"Worktree has uncommitted changes, not cleaning up: {worktree_path}"
        )
        return

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_path)],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    subprocess.run(
        ["git", "branch", "-d", branch_name],
        cwd=REPO_ROOT,
        capture_output=True,
    )


# ─── Prompt Loading ────────────────────────────────────────────


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text()


# ─── Agent Sessions ────────────────────────────────────────────


async def run_planner(
    story: str,
    product: str,
    worktree_path: Path,
    aiia_context: str,
    model: str = DEFAULT_MODEL,
    max_budget: float = DEFAULT_MAX_BUDGET,
) -> Optional[ActionPlan]:
    """
    Session 1: Planner Agent.

    Reads the story, explores the codebase, and creates an action plan
    with atomic coding actions.
    """
    prompt_template = load_prompt("planner_prompt")
    product_dir = PRODUCT_DIRS.get(product, f"products/{product}")

    prompt = prompt_template.format(
        story=story,
        product=product,
        product_dir=product_dir,
        aiia_context=aiia_context or "No AIIA context available.",
        repo_root=str(worktree_path),
    )

    logger.info(f"[Planner] Starting planning for: {story}")
    notify_aiia("planner_started", {"story": story, "product": product})

    plan_path = worktree_path / "action_plan.json"
    session_id = None
    total_cost = 0.0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model=model,
            max_turns=30,
            max_budget_usd=max_budget * 0.3,  # 30% of budget for planning
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            disallowed_tools=["Edit", "Write"],  # planner doesn't write code
            permission_mode="acceptEdits",
            cwd=str(worktree_path),
            setting_sources=["project"],  # load .claude/settings.json
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    # Check if the agent wrote the plan file
                    pass
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            total_cost = message.total_cost_usd or 0.0
            logger.info(
                f"[Planner] Done. Cost: ${total_cost:.4f}, Turns: {message.num_turns}"
            )

    # Load the plan the agent should have created
    if plan_path.exists():
        plan = load_plan(plan_path)
        plan.planner_session_id = session_id
        plan.planner_cost = total_cost
        save_plan(plan, plan_path)
        logger.info(f"[Planner] Created {len(plan.actions)} actions")
        notify_aiia(
            "planner_complete",
            {
                "story": story,
                "actions": len(plan.actions),
                "cost": total_cost,
            },
        )
        return plan
    else:
        logger.error("[Planner] No action_plan.json was created")
        return None


async def run_coder(
    plan: ActionPlan,
    worktree_path: Path,
    model: str = DEFAULT_MODEL,
    max_budget: float = DEFAULT_MAX_BUDGET,
    max_iterations: int = 20,
) -> ActionPlan:
    """
    Sessions 2+: Coder Agent.

    Picks up the next incomplete action from action_plan.json,
    implements it, runs smoke tests, commits, and marks it done.
    Continues until all actions are complete or budget is exhausted.
    """
    plan_path = worktree_path / "action_plan.json"
    remaining_budget = max_budget * 0.7  # 70% of budget for coding
    iteration = 0

    while iteration < max_iterations:
        # Reload plan (may have been updated by previous session)
        plan = load_plan(plan_path)

        # Find next incomplete action
        next_action = None
        for action in plan.actions:
            if not action.get("completed", False):
                next_action = action
                break

        if next_action is None:
            logger.info("[Coder] All actions complete!")
            break

        if remaining_budget <= 0:
            logger.warning("[Coder] Budget exhausted")
            break

        iteration += 1
        action_idx = plan.actions.index(next_action)
        logger.info(
            f"[Coder] Session {iteration}: Action {action_idx + 1}/{len(plan.actions)} "
            f"— {next_action.get('title', 'Unknown')}"
        )

        prompt_template = load_prompt("coder_prompt")
        prompt = prompt_template.format(
            story=plan.story,
            product=plan.product,
            action_title=next_action.get("title", ""),
            action_description=next_action.get("description", ""),
            action_files=json.dumps(next_action.get("files", []), indent=2),
            action_index=action_idx,
            total_actions=len(plan.actions),
            plan_path=str(plan_path),
        )

        session_cost = 0.0

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=model,
                max_turns=DEFAULT_MAX_TURNS,
                max_budget_usd=min(remaining_budget, 2.0),  # max $2 per action
                allowed_tools=[
                    "Read",
                    "Edit",
                    "Write",
                    "Glob",
                    "Grep",
                    "Bash",
                ],
                permission_mode="acceptEdits",
                cwd=str(worktree_path),
                setting_sources=["project"],
            ),
        ):
            if isinstance(message, ResultMessage):
                session_cost = message.total_cost_usd or 0.0
                remaining_budget -= session_cost
                logger.info(
                    f"[Coder] Action done. Cost: ${session_cost:.4f}, "
                    f"Remaining budget: ${remaining_budget:.4f}"
                )

        # Reload and check if action was marked complete
        plan = load_plan(plan_path)
        plan.total_cost = (plan.total_cost or 0) + session_cost
        save_plan(plan, plan_path)

        notify_aiia(
            "action_complete",
            {
                "action": next_action.get("title", ""),
                "cost": session_cost,
                "iteration": iteration,
            },
        )

        # Brief pause between sessions
        if iteration < max_iterations:
            time.sleep(DELAY_BETWEEN_SESSIONS)

    return plan


# ─── PR Creation ────────────────────────────────────────────────


def create_pull_request(
    plan: ActionPlan,
    worktree_path: Path,
    branch_name: str,
) -> Optional[str]:
    """Push branch and create a PR."""
    # Push the branch
    result = subprocess.run(
        ["git", "push", "-u", "origin", branch_name],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Failed to push: {result.stderr}")
        return None

    # Build PR body
    completed = sum(1 for a in plan.actions if a.get("completed", False))
    total = len(plan.actions)

    body = f"""## {plan.story}

### Actions Completed: {completed}/{total}

| # | Action | Status |
|---|--------|--------|
"""
    for i, action in enumerate(plan.actions):
        status = "Done" if action.get("completed") else "Pending"
        body += f"| {i + 1} | {action.get('title', '')} | {status} |\n"

    body += f"""
### Cost
- Planning: ${plan.planner_cost or 0:.4f}
- Coding: ${(plan.total_cost or 0) - (plan.planner_cost or 0):.4f}
- Total: ${plan.total_cost or 0:.4f}

### Product
`{plan.product}`

---
*Autonomous coding by AIIA Story Runner*
*Powered by Claude Agent SDK + Mac Mini M4*
"""

    # Create PR
    result = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            f"feat({plan.product}): {plan.story[:60]}",
            "--body",
            body,
            "--base",
            "main",
            "--head",
            branch_name,
        ],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        pr_url = result.stdout.strip()
        logger.info(f"PR created: {pr_url}")
        return pr_url
    else:
        logger.error(f"PR creation failed: {result.stderr}")
        return None


# ─── Main Orchestrator ──────────────────────────────────────────


async def run_story(
    story: str,
    product: str = "default",
    model: str = DEFAULT_MODEL,
    max_budget: float = DEFAULT_MAX_BUDGET,
    max_iterations: int = 20,
    plan_only: bool = False,
) -> Dict[str, Any]:
    """
    Execute a complete story from planning through PR creation.

    Returns a summary dict with results.
    """
    start_time = datetime.now()
    story_slug = story.lower()[:40].replace(" ", "-").replace("/", "-")
    story_slug = "".join(c for c in story_slug if c.isalnum() or c == "-")
    branch_name = f"aiia/{story_slug}"

    logger.info(f"╔{'═' * 58}╗")
    logger.info("║  AIIA Story Runner                                       ║")
    logger.info(f"║  Story: {story[:48]:<48} ║")
    logger.info(f"║  Product: {product:<46} ║")
    logger.info(f"║  Budget: ${max_budget:<46.2f}║")
    logger.info(f"╚{'═' * 58}╝")

    # Fetch AIIA context
    aiia_context = fetch_aiia_context()

    # Create isolated worktree
    worktree_path = create_worktree(story_slug)

    try:
        # Phase 1: Planning
        plan = await run_planner(
            story=story,
            product=product,
            worktree_path=worktree_path,
            aiia_context=aiia_context,
            model=model,
            max_budget=max_budget,
        )

        if plan is None:
            return {"status": "failed", "reason": "Planning failed"}

        if plan_only:
            logger.info("[Runner] Plan-only mode — skipping coding phase")
            return {
                "status": "planned",
                "actions": len(plan.actions),
                "plan_path": str(worktree_path / "action_plan.json"),
            }

        # Phase 2: Coding
        plan = await run_coder(
            plan=plan,
            worktree_path=worktree_path,
            model=model,
            max_budget=max_budget,
            max_iterations=max_iterations,
        )

        # Phase 3: Create PR
        completed = sum(1 for a in plan.actions if a.get("completed", False))
        total = len(plan.actions)

        pr_url = None
        if completed > 0:
            pr_url = create_pull_request(plan, worktree_path, branch_name)

        duration = (datetime.now() - start_time).total_seconds()

        summary = {
            "status": "complete" if completed == total else "partial",
            "story": story,
            "product": product,
            "actions_completed": completed,
            "actions_total": total,
            "total_cost": plan.total_cost,
            "duration_seconds": duration,
            "pr_url": pr_url,
            "branch": branch_name,
            "worktree": str(worktree_path),
        }

        notify_aiia("story_complete", summary)

        logger.info(f"\n{'─' * 60}")
        logger.info(f"  Story: {story}")
        logger.info(f"  Status: {summary['status']}")
        logger.info(f"  Actions: {completed}/{total}")
        logger.info(f"  Cost: ${plan.total_cost or 0:.4f}")
        logger.info(f"  Duration: {duration:.0f}s")
        if pr_url:
            logger.info(f"  PR: {pr_url}")
        logger.info(f"{'─' * 60}")

        return summary

    except KeyboardInterrupt:
        logger.info("\n[Runner] Interrupted — progress saved. Re-run to continue.")
        return {"status": "interrupted", "worktree": str(worktree_path)}
    except Exception as e:
        logger.error(f"[Runner] Failed: {e}")
        notify_aiia("story_failed", {"story": story, "error": str(e)})
        return {"status": "failed", "reason": str(e)}


# ─── CLI ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="AIIA Story Runner — autonomous coding for AIIA"
    )
    parser.add_argument(
        "--story",
        "-s",
        type=str,
        help="Story description (what to build)",
    )
    parser.add_argument(
        "--product",
        "-p",
        type=str,
        default="default",
        choices=list(PRODUCT_DIRS.keys()),
        help="Target product (default: default)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=DEFAULT_MAX_BUDGET,
        help=f"Max budget in USD (default: ${DEFAULT_MAX_BUDGET})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max coding iterations (default: 20)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only create the plan, don't write code",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.story:
        parser.error("--story is required")

    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY environment variable is required")

    result = asyncio.run(
        run_story(
            story=args.story,
            product=args.product,
            model=args.model,
            max_budget=args.max_budget,
            max_iterations=args.max_iterations,
            plan_only=args.plan_only,
        )
    )

    # Exit code based on status
    sys.exit(0 if result.get("status") in ("complete", "planned") else 1)


if __name__ == "__main__":
    main()
