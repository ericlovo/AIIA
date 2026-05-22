from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .subprocess_pool import SubprocessPool
from .verification import Verifier

if TYPE_CHECKING:
    from .git_ops import GitOps

logger = logging.getLogger("aiia.execution.strategies")

REPO_PATH = os.environ.get(
    "AIIA_REPO_PATH",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


@dataclass
class ExecutionResult:
    success: bool
    output: str
    files_changed: list[str]
    error: str | None = None

    def __post_init__(self):
        if len(self.output) > 2000:
            self.output = self.output[:1997] + "..."


class ExecutionStrategy(ABC):
    max_timeout: int = 120
    has_verification: bool = True

    @abstractmethod
    async def execute(self, action: dict) -> ExecutionResult: ...

    @abstractmethod
    async def verify(self, action: dict, result: ExecutionResult) -> bool: ...


class DirectFixStrategy(ExecutionStrategy):
    max_timeout = 120
    has_verification = True

    def __init__(self, pool: SubprocessPool, repo_path: str, verifier: Verifier):
        self._pool = pool
        self._repo_path = repo_path
        self._verifier = verifier

    async def execute(self, action: dict) -> ExecutionResult:
        action_type = action.get("type", "")
        if action_type == "lint_fix":
            return await self._fix_lint(action)
        elif action_type == "security_fix" and action.get("dep_bump"):
            return await self._fix_dep(action)
        return ExecutionResult(
            success=False,
            output="",
            files_changed=[],
            error=f"DirectFix doesn't handle type '{action_type}'",
        )

    async def verify(self, action: dict, result: ExecutionResult) -> bool:
        v = await self._verifier.verify(action)
        return v.verified is True

    async def _fix_lint(self, action: dict) -> ExecutionResult:
        files = action.get("files_affected", [])
        if not files:
            return ExecutionResult(
                success=False,
                output="",
                files_changed=[],
                error="No files_affected for lint fix",
            )

        fix_result = await self._pool.run(
            ["ruff", "check", "--fix"] + files,
            cwd=self._repo_path,
            timeout=30,
        )
        fmt_result = await self._pool.run(
            ["ruff", "format"] + files,
            cwd=self._repo_path,
            timeout=30,
        )

        combined = fix_result.stdout + fmt_result.stdout
        return ExecutionResult(
            success=True,
            output=combined,
            files_changed=files,
        )

    async def _fix_dep(self, action: dict) -> ExecutionResult:
        package = action.get("package", "")
        if not package:
            return ExecutionResult(
                success=False,
                output="",
                files_changed=[],
                error="No package specified for dep bump",
            )

        result = await self._pool.run(
            ["pip", "install", "--upgrade", package],
            cwd=self._repo_path,
            timeout=60,
        )
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            files_changed=["requirements.txt"],
            error=result.stderr if result.returncode != 0 else None,
        )


class ClaudeCodeStrategy(ExecutionStrategy):
    max_timeout = 600
    has_verification = True

    def __init__(
        self,
        pool: SubprocessPool,
        repo_path: str,
        verifier: Verifier,
        branch_prefix: str = "aiia/",
        git_ops: GitOps | None = None,
    ):
        self._pool = pool
        self._repo_path = repo_path
        self._verifier = verifier
        self._branch_prefix = branch_prefix
        self._git = git_ops

    async def execute(self, action: dict) -> ExecutionResult:
        action_type = action.get("type", "unknown")
        action_id = action.get("id", "unknown")[:8]
        original_branch = None

        # Git branch management
        if self._git:
            original_branch = await self._git.current_branch()
            branch_name = f"{action_type}/{action_id}"
            await self._git.create_branch(branch_name)

        # Build and execute prompt
        prompt = self._build_prompt(action)
        try:
            result = await self._pool.run_claude_code(
                prompt=prompt,
                cwd=self._repo_path,
                timeout=self.max_timeout,
            )
        except Exception:
            # On failure, clean up branch
            if self._git and original_branch:
                await self._git.abort_and_cleanup()
            raise

        # Get changed files from git if available
        if self._git:
            files_changed = await self._git.get_changed_files()
        else:
            files_changed = self._parse_changed_files(result.stdout)

        if result.returncode != 0 and self._git and original_branch:
            await self._git.abort_and_cleanup()

        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            files_changed=files_changed,
            error=result.stderr if result.returncode != 0 else None,
        )

    async def verify(self, action: dict, result: ExecutionResult) -> bool:
        v = await self._verifier.verify(action)
        return v.verified is True

    def _build_prompt(self, action: dict) -> str:
        files = action.get("files_affected", [])
        files_str = "\n".join(f"  - {f}" for f in files) if files else "  (not specified)"
        proposed = action.get("proposed_fix", "")

        prompt = (
            f"Fix this issue in the AIIA codebase:\n\n"
            f"Title: {action.get('title', '')}\n"
            f"Description: {action.get('description', '')}\n\n"
            f"Files affected:\n{files_str}\n"
        )
        if proposed:
            prompt += f"\nSuggested approach: {proposed}\n"

        prompt += (
            "\nConstraints:\n"
            "- Only modify the files listed above unless absolutely necessary\n"
            "- Do NOT create new files unless required\n"
            "- Run tests after making changes\n"
            "- Do not modify local_brain/ unless the fix is platform-level\n"
            "- Follow code style: black 88, isort black profile\n"
            "- Do not add unnecessary docstrings or comments\n"
            "- Summarize what you changed at the end\n"
        )
        return prompt

    def _parse_changed_files(self, output: str) -> list[str]:
        files = []
        for line in output.splitlines():
            stripped = line.strip()
            for prefix in ("Modified ", "Created ", "Edit: ", "Write: "):
                if stripped.startswith(prefix):
                    path = stripped[len(prefix) :].strip()
                    if path:
                        files.append(path)
                    break
        return files


class BranchPrepStrategy(ExecutionStrategy):
    """Creates a branch with context for human-in-the-loop coding.

    Does NOT write code or make fixes. It:
    1. Creates an aiia/{story} branch
    2. Writes .aiia-context.md with story details, proposed fix, affected files
    3. Commits the context file
    4. Stops — Claude + Eric pick it up from here
    """

    max_timeout = 30
    has_verification = False

    def __init__(self, git_ops: GitOps, repo_path: str):
        self._git = git_ops
        self._repo_path = repo_path

    async def execute(self, action: dict) -> ExecutionResult:
        import os

        action_id = action.get("id", "unknown")[:8]
        title = action.get("title", "untitled")
        description = action.get("description", "")
        proposed_fix = action.get("proposed_fix", "")
        files = action.get("files_affected", [])
        story_id = action.get("story_id", "")
        story_step = action.get("story_step", "")
        story_step_total = action.get("story_step_total", "")

        # Create branch
        branch_name = f"prep/{action_id}-{_slugify(title)}"
        created = await self._git.create_branch(branch_name)
        if not created:
            return ExecutionResult(
                success=False,
                output="",
                files_changed=[],
                error=f"Failed to create branch aiia/{branch_name}",
            )

        # Write context file
        files_section = "\n".join(f"- `{f}`" for f in files) if files else "- (not specified)"
        step_info = f"Step {story_step}/{story_step_total}" if story_step else ""

        context = (
            f"# AIIA Branch Context\n\n"
            f"**Story:** {story_id}\n"
            f"**Action:** {action_id}\n"
            f"{f'**Progress:** {step_info}' + chr(10) if step_info else ''}"
            f"**Status:** Ready for human-in-the-loop coding with Claude\n\n"
            f"## What needs to happen\n\n"
            f"**{title}**\n\n"
            f"{description}\n\n"
            f"## Proposed approach\n\n"
            f"{proposed_fix if proposed_fix else '(no specific approach suggested — use your judgment)'}\n\n"
            f"## Files to look at\n\n"
            f"{files_section}\n\n"
            f"---\n"
            f"*Generated by AIIA execution engine. "
            f"Delete this file before merging.*\n"
        )

        context_path = os.path.join(self._repo_path, ".aiia-context.md")
        try:
            with open(context_path, "w") as f:
                f.write(context)
        except OSError as e:
            await self._git.abort_and_cleanup()
            return ExecutionResult(
                success=False,
                output="",
                files_changed=[],
                error=f"Failed to write context file: {e}",
            )

        # Stage and commit the context file
        await self._git.stage_files([".aiia-context.md"])
        sha = await self._git.commit(
            f"prep: {title}\n\nBranch ready for human-in-the-loop coding.\nStory: {story_id}"
        )

        # Return to main so the branch sits ready
        await self._git.return_to_main()

        branch_full = f"aiia/{branch_name}"
        return ExecutionResult(
            success=True,
            output=(
                f"Branch `{branch_full}` created and ready.\n"
                f"Commit: {sha or 'unknown'}\n"
                f"Context file: .aiia-context.md\n\n"
                f"To start working:\n"
                f"  git checkout {branch_full}\n"
                f"  # Then work with Claude on the changes"
            ),
            files_changed=[".aiia-context.md"],
        )

    async def verify(self, action: dict, result: ExecutionResult) -> bool:
        return True


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a git-branch-safe slug."""
    import re

    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-")


class CommitStrategy(ExecutionStrategy):
    max_timeout = 30
    has_verification = True

    def __init__(self, git_ops: GitOps):
        self._git = git_ops

    async def execute(self, action: dict) -> ExecutionResult:
        parent_title = action.get("title", "AIIA automated fix")
        # Strip "Verify: " prefix if chained from verification
        clean_title = parent_title
        for prefix in ("Verify: ", "Verify tests: ", "Verify security: "):
            if clean_title.startswith(prefix):
                clean_title = clean_title[len(prefix) :]
                break

        msg = f"fix: {clean_title}\n\nAutomated by AIIA execution engine"

        await self._git.stage_all()
        commit_hash = await self._git.commit(msg)

        if commit_hash:
            return ExecutionResult(
                success=True,
                output=f"Committed: {commit_hash}",
                files_changed=[],
            )
        return ExecutionResult(
            success=True,
            output="Nothing to commit — working tree clean",
            files_changed=[],
        )

    async def verify(self, action: dict, result: ExecutionResult) -> bool:
        has_changes = await self._git.has_uncommitted_changes()
        return not has_changes


def select_strategy(
    action: dict,
    direct: DirectFixStrategy,
    claude: ClaudeCodeStrategy,
    commit: CommitStrategy | None = None,
    branch_prep: BranchPrepStrategy | None = None,
) -> ExecutionStrategy:
    action_type = action.get("type", "")
    if action_type == "lint_fix":
        return direct
    if action_type == "security_fix" and action.get("dep_bump"):
        return direct
    if action_type == "commit" and commit:
        return commit
    if action_type == "branch_prep" and branch_prep:
        return branch_prep
    return claude
