"""Git operations helper for AIIA's execution engine.

Safe, local-only git operations. Never pushes, force-resets, or rebases.
All methods are async and use SubprocessPool for execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .subprocess_pool import SubprocessPool

logger = logging.getLogger("aiia.execution.git")

BRANCH_PREFIX = "aiia/"


@dataclass
class GitStatus:
    branch: str = "unknown"
    changed_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    has_changes: bool = False


class GitOps:
    """Safe git operations for AIIA execution engine.

    All operations are local-only — no push, no force-reset, no rebase.
    Methods return sensible defaults on error and never raise.
    """

    def __init__(self, repo_path: str, pool: SubprocessPool):
        self._repo_path = repo_path
        self._pool = pool

    async def _git(self, args: list[str], timeout: float = 30) -> tuple[int, str, str]:
        """Run a git command. Returns (returncode, stdout, stderr)."""
        try:
            result = await self._pool.run(
                ["git"] + args, cwd=self._repo_path, timeout=timeout
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            logger.error(f"git {args[0]} failed: {e}")
            return 1, "", str(e)

    # ── Branch operations ──────────────────────────────────────────

    async def current_branch(self) -> str:
        """Return the current branch name, or 'unknown' on error."""
        rc, out, _ = await self._git(["rev-parse", "--abbrev-ref", "HEAD"])
        return out if rc == 0 and out else "unknown"

    async def create_branch(self, name: str) -> bool:
        """Create and checkout a new aiia/ prefixed branch."""
        branch = (
            f"{BRANCH_PREFIX}{name}" if not name.startswith(BRANCH_PREFIX) else name
        )
        rc, _, err = await self._git(["checkout", "-b", branch])
        if rc != 0:
            logger.error(f"Failed to create branch {branch}: {err}")
            return False
        logger.info(f"Created branch {branch}")
        return True

    async def checkout(self, branch: str) -> bool:
        """Checkout an existing branch."""
        rc, _, err = await self._git(["checkout", branch])
        if rc != 0:
            logger.error(f"Failed to checkout {branch}: {err}")
            return False
        return True

    async def return_to_main(self) -> bool:
        """Return to main branch. Tries 'main' first, then 'master'."""
        if await self.checkout("main"):
            return True
        return await self.checkout("master")

    # ── Diff & status ──────────────────────────────────────────────

    async def get_diff(self, staged: bool = False) -> str:
        """Get diff output. If staged=True, shows staged changes only."""
        args = ["diff"]
        if staged:
            args.append("--cached")
        rc, out, _ = await self._git(args, timeout=60)
        return out if rc == 0 else ""

    async def get_changed_files(self) -> list[str]:
        """List files with uncommitted changes (staged + unstaged)."""
        rc, out, _ = await self._git(["status", "--porcelain", "--no-renames"])
        if rc != 0 or not out:
            return []
        files = []
        for line in out.splitlines():
            # porcelain format: XY filename
            if len(line) > 3:
                files.append(line[3:].strip())
        return files

    async def get_status(self) -> GitStatus:
        """Get full git status."""
        branch = await self.current_branch()
        rc, out, _ = await self._git(["status", "--porcelain", "--no-renames"])
        if rc != 0 or not out:
            return GitStatus(branch=branch)

        changed = []
        staged = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            index_status = line[0]
            worktree_status = line[1]
            filename = line[3:].strip()

            if index_status != " " and index_status != "?":
                staged.append(filename)
            if worktree_status != " " or index_status == "?":
                changed.append(filename)

        return GitStatus(
            branch=branch,
            changed_files=changed,
            staged_files=staged,
            has_changes=bool(changed or staged),
        )

    async def has_uncommitted_changes(self) -> bool:
        """Check if there are any uncommitted changes."""
        rc, out, _ = await self._git(["status", "--porcelain"])
        return rc == 0 and bool(out.strip())

    # ── Staging ────────────────────────────────────────────────────

    async def stage_files(self, files: list[str]) -> bool:
        """Stage specific files."""
        if not files:
            return True
        rc, _, err = await self._git(["add", "--"] + files)
        if rc != 0:
            logger.error(f"Failed to stage files: {err}")
            return False
        return True

    async def stage_all(self) -> bool:
        """Stage all changes."""
        rc, _, err = await self._git(["add", "-A"])
        if rc != 0:
            logger.error(f"Failed to stage all: {err}")
            return False
        return True

    # ── Commit ─────────────────────────────────────────────────────

    async def commit(self, message: str) -> Optional[str]:
        """Create a commit. Returns the short SHA on success, None on failure."""
        rc, _, err = await self._git(["commit", "-m", message])
        if rc != 0:
            logger.error(f"Commit failed: {err}")
            return None
        # Get the SHA of the new commit
        rc, sha, _ = await self._git(["rev-parse", "--short", "HEAD"])
        if rc == 0 and sha:
            logger.info(f"Committed {sha}: {message[:60]}")
            return sha
        return None

    # ── Cleanup ────────────────────────────────────────────────────

    async def abort_and_cleanup(self) -> bool:
        """Discard all uncommitted changes and return to main.

        Uses git checkout -- . to discard modifications and
        git clean -fd to remove untracked files, then returns to main.
        """
        current = await self.current_branch()
        errors = []

        # Discard tracked file changes
        rc, _, err = await self._git(["checkout", "--", "."])
        if rc != 0:
            errors.append(f"checkout: {err}")

        # Remove untracked files
        rc, _, err = await self._git(["clean", "-fd"])
        if rc != 0:
            errors.append(f"clean: {err}")

        # Return to main if on an aiia/ branch
        if current.startswith(BRANCH_PREFIX):
            if not await self.return_to_main():
                errors.append("failed to return to main")
            else:
                # Delete the aiia branch we were on
                rc, _, err = await self._git(["branch", "-D", current])
                if rc != 0:
                    errors.append(f"branch delete: {err}")

        if errors:
            logger.warning(f"Cleanup had errors: {'; '.join(errors)}")
            return False

        logger.info("Cleanup complete")
        return True

    # ── Read-only queries ──────────────────────────────────────────

    async def log(self, n: int = 10, since: Optional[str] = None) -> list[dict]:
        """Get recent commit log entries."""
        args = ["log", f"-{n}", "--format=%H|%s|%an|%ai", "--no-merges"]
        if since:
            args.append(f"--since={since}")
        rc, out, _ = await self._git(args)
        if rc != 0 or not out:
            return []

        commits = []
        for line in out.splitlines():
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            commits.append(
                {
                    "sha": parts[0][:12],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                }
            )
        return commits

    async def file_diff(self, filepath: str) -> str:
        """Get diff for a single file."""
        rc, out, _ = await self._git(["diff", "--", filepath])
        return out if rc == 0 else ""
