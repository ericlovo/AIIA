"""Approval-gated, isolated Git workspaces for Agent Studio assignments."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repository_tools import repo_mount, repo_write_eligibility

logger = logging.getLogger("aiia.git_workspaces")

WORKSPACE_DATA_FILE = Path(__file__).parent / "git_workspace_data.json"
WORKTREE_ROOT = Path(os.getenv("AIIA_WORKTREE_ROOT", Path.home() / ".aiia" / "worktrees"))
MAX_WORKSPACES = 250


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str, max_length: int = 42) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "assignment"


def _run_git(repo: Path, *args: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class GitWorkspaceRegistry:
    """Persist proposals and create worktrees only after explicit approval."""

    def __init__(
        self,
        data_file: Path | None = None,
        worktree_root: Path | None = None,
    ):
        self.data_file = data_file or WORKSPACE_DATA_FILE
        self.worktree_root = worktree_root or WORKTREE_ROOT
        self.workspaces: list[dict[str, Any]] = []
        self.load()

    def list(self) -> list[dict[str, Any]]:
        return sorted(self.workspaces, key=lambda item: item["created_at"], reverse=True)

    def get(self, workspace_id: str) -> dict[str, Any] | None:
        return next((item for item in self.workspaces if item["id"] == workspace_id), None)

    def for_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.list() if item["assignment_id"] == assignment_id),
            None,
        )

    def propose(self, assignment: dict[str, Any], agent: dict[str, Any]) -> dict[str, Any]:
        existing = self.for_assignment(assignment["id"])
        if existing:
            return existing
        if assignment.get("status") != "completed":
            raise ValueError("assignment_not_completed")
        if "Git workspace" not in set(agent.get("tools", [])):
            raise ValueError("git_workspace_tool_required")

        repo_id = str(agent.get("repo_id", ""))
        eligible, reason = repo_write_eligibility(repo_id)
        if not eligible:
            raise ValueError(reason)

        workspace_id = f"gws_{uuid.uuid4().hex[:12]}"
        branch = f"aiia/{repo_id}/{_slug(assignment['title'])}-{workspace_id[-6:]}"
        now = _now()
        workspace = {
            "id": workspace_id,
            "assignment_id": assignment["id"],
            "agent_id": agent["id"],
            "repo_id": repo_id,
            "title": assignment["title"],
            "status": "pending",
            "branch": branch,
            "base_ref": "",
            "path": "",
            "git_status": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
            "approved_at": None,
            "events": [
                {
                    "at": now,
                    "action": "requested",
                    "detail": "Awaiting explicit workspace approval",
                }
            ],
        }
        self.workspaces.insert(0, workspace)
        self.workspaces = self.workspaces[:MAX_WORKSPACES]
        self.save()
        return workspace

    def approve(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.get(workspace_id)
        if not workspace:
            raise ValueError("git_workspace_not_found")
        if workspace["status"] == "ready":
            return self.refresh(workspace_id)
        if workspace["status"] != "pending":
            raise ValueError("git_workspace_not_approvable")

        repo_id = workspace["repo_id"]
        eligible, reason = repo_write_eligibility(repo_id)
        if not eligible:
            raise ValueError(reason)
        repo = repo_mount(repo_id)
        if not repo:
            raise ValueError("repository_not_mounted")

        path = self.worktree_root / repo_id / workspace_id
        if path.exists():
            raise ValueError("git_workspace_path_exists")

        base_ref = self._base_ref(repo)
        workspace["status"] = "preparing"
        workspace["base_ref"] = base_ref
        workspace["updated_at"] = _now()
        self.save()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            result = _run_git(
                repo,
                "worktree",
                "add",
                "-b",
                workspace["branch"],
                str(path),
                base_ref,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._fail(workspace, f"git_worktree_failed: {exc}")

        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "git worktree add failed"
            return self._fail(workspace, error[:2_000])

        now = _now()
        workspace.update(
            {
                "status": "ready",
                "path": str(path),
                "git_status": self._status(path),
                "error": "",
                "approved_at": now,
                "updated_at": now,
            }
        )
        workspace["events"].append(
            {
                "at": now,
                "action": "approved",
                "detail": f"Created from {base_ref}",
            }
        )
        self.save()
        return workspace

    def refresh(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.get(workspace_id)
        if not workspace:
            raise ValueError("git_workspace_not_found")
        path = Path(workspace.get("path", ""))
        if workspace["status"] == "ready" and path.is_dir():
            workspace["git_status"] = self._status(path)
            workspace["updated_at"] = _now()
            self.save()
        return workspace

    @staticmethod
    def _base_ref(repo: Path) -> str:
        remote_head = _run_git(
            repo,
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
        )
        if remote_head.returncode == 0 and remote_head.stdout.strip():
            return remote_head.stdout.strip()
        for candidate in ("main", "master"):
            result = _run_git(repo, "show-ref", "--verify", f"refs/heads/{candidate}")
            if result.returncode == 0:
                return candidate
        return "HEAD"

    @staticmethod
    def _status(path: Path) -> str:
        result = _run_git(path, "status", "--short", "--branch")
        if result.returncode != 0:
            return "unavailable"
        return result.stdout.strip() or "clean"

    def _fail(self, workspace: dict[str, Any], error: str) -> dict[str, Any]:
        now = _now()
        workspace.update({"status": "failed", "error": error, "updated_at": now})
        workspace["events"].append({"at": now, "action": "failed", "detail": error})
        self.save()
        return workspace

    def save(self) -> None:
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self.data_file.write_text(json.dumps({"workspaces": self.workspaces}, indent=2))
        except OSError as exc:
            logger.error("Could not save Git workspaces: %s", exc)

    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            payload = json.loads(self.data_file.read_text())
            self.workspaces = payload.get("workspaces", [])[:MAX_WORKSPACES]
            for workspace in self.workspaces:
                workspace.setdefault("events", [])
                workspace.setdefault("git_status", "")
                if workspace.get("status") == "preparing":
                    workspace["status"] = "failed"
                    workspace["error"] = "interrupted_by_restart"
                    workspace["updated_at"] = _now()
            self.save()
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load Git workspaces: %s", exc)
