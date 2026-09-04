"""Approval-gated Git writes against an approved Agent Studio worktree.

Agents and the UI may *propose* a write. A human must *approve* before anything
touches the isolated worktree. Operations never run against the source checkout.

Live ops (executed only after human approve, only on the worktree path):
  - write_file: relative path + content; reject path escape
  - run_tests: fixed command allowlist (pytest -q, npm test,
    npm test -- --watchAll=false). No free shell from the model.
  - commit: message required; git add specific paths + commit inside
    the worktree only

Deferred ops (propose is rejected with these exact reasons):
  - push -> op_deferred:push
  - open_pr -> op_deferred:open_pr
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .git_workspace_registry import GitWorkspaceRegistry
from .repository_tools import repo_mount

logger = logging.getLogger("aiia.git_writes")

WRITE_DATA_FILE = Path(__file__).parent / "git_write_data.json"
MAX_WRITES = 250
MAX_FILE_BYTES = 1_000_000
MAX_TEST_OUTPUT = 20_000
TEST_TIMEOUT_SECONDS = 120.0

LIVE_OPS = frozenset({"write_file", "run_tests", "commit"})
DEFERRED_OPS = {
    "push": "op_deferred:push",
    "open_pr": "op_deferred:open_pr",
}
READY_WORKSPACE_STATUSES = frozenset({"ready", "approved"})
TEST_COMMANDS: dict[str, list[str]] = {
    "pytest -q": ["pytest", "-q", "--rootdir=.", "--noconftest"],
    "npm test": ["npm", "test"],
    "npm test -- --watchAll=false": ["npm", "test", "--", "--watchAll=false"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(repo: Path, *args: str, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def safe_worktree_path(worktree: Path, relative: str) -> Path:
    """Resolve a worktree-relative path or raise ``path_escape``."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("path_escape")
    relative = relative.strip().replace("\\", "/")
    if relative.endswith("/"):
        raise ValueError("path_escape")
    if relative.startswith("/") or Path(relative).is_absolute():
        raise ValueError("path_escape")
    if len(relative) >= 2 and relative[1] == ":":
        raise ValueError("path_escape")
    parts = Path(relative).parts
    if not parts or any(part in {"..", "", ".", ".git"} for part in parts):
        raise ValueError("path_escape")
    worktree_resolved = worktree.resolve()
    target = (worktree_resolved / relative).resolve()
    try:
        target.relative_to(worktree_resolved)
    except ValueError as exc:
        raise ValueError("path_escape") from exc
    if target == worktree_resolved:
        raise ValueError("path_escape")
    return target


class GitWriteRegistry:
    """Persist write proposals and execute allowlisted ops only after approval."""

    def __init__(
        self,
        data_file: Path | None = None,
        workspace_registry: GitWorkspaceRegistry | None = None,
    ):
        self.data_file = data_file or WRITE_DATA_FILE
        self.workspace_registry = workspace_registry
        self.writes: list[dict[str, Any]] = []
        self.load()

    def list(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        items = self.writes
        if workspace_id:
            items = [item for item in items if item["workspace_id"] == workspace_id]
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def get(self, write_id: str) -> dict[str, Any] | None:
        return next((item for item in self.writes if item["id"] == write_id), None)

    def propose(
        self,
        workspace_id: str,
        op: str,
        payload: dict[str, Any] | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        if op in DEFERRED_OPS:
            raise ValueError(DEFERRED_OPS[op])
        if op not in LIVE_OPS:
            raise ValueError("unknown_op")

        workspace, worktree = self._ready_worktree(workspace_id)
        self._validate_payload(op, payload, worktree)

        now = _now()
        write = {
            "id": f"gwr_{uuid.uuid4().hex[:12]}",
            "workspace_id": workspace["id"],
            "assignment_id": workspace.get("assignment_id", ""),
            "op": op,
            "title": (title or self._default_title(op, payload)).strip()[:200],
            "status": "pending",
            "payload": payload,
            "result": {},
            "error": "",
            "created_at": now,
            "updated_at": now,
            "approved_at": None,
            "events": [
                {
                    "at": now,
                    "action": "requested",
                    "detail": f"Awaiting approval to {op} on worktree",
                }
            ],
        }
        self.writes.insert(0, write)
        self.writes = self.writes[:MAX_WRITES]
        self.save()
        return write

    def approve(self, write_id: str) -> dict[str, Any]:
        write = self.get(write_id)
        if not write:
            raise ValueError("git_write_not_found")
        if write["status"] in {"completed", "failed"}:
            return write
        if write["status"] != "pending":
            raise ValueError("git_write_not_approvable")

        workspace, worktree = self._ready_worktree(write["workspace_id"])
        self._validate_payload(write["op"], write["payload"], worktree)

        now = _now()
        write["status"] = "approved"
        write["approved_at"] = now
        write["updated_at"] = now
        write["events"].append(
            {"at": now, "action": "approved", "detail": "Executing allowlisted op"}
        )
        self.save()

        try:
            result = self._execute(write["op"], write["payload"], worktree)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._fail(write, f"git_write_failed: {exc}")
        except ValueError as exc:
            return self._fail(write, str(exc))

        now = _now()
        write.update(
            {
                "status": "completed",
                "result": result,
                "error": "",
                "updated_at": now,
            }
        )
        write["events"].append({"at": now, "action": "completed", "detail": write["op"]})
        self.save()
        self._refresh_workspace(workspace["id"])
        return write

    def reject(self, write_id: str, reason: str = "") -> dict[str, Any]:
        write = self.get(write_id)
        if not write:
            raise ValueError("git_write_not_found")
        if write["status"] != "pending":
            raise ValueError("git_write_not_rejectable")
        now = _now()
        write.update(
            {
                "status": "rejected",
                "error": reason[:2_000],
                "updated_at": now,
            }
        )
        write["events"].append(
            {"at": now, "action": "rejected", "detail": reason[:2_000] or "Rejected"}
        )
        self.save()
        return write

    def _ready_worktree(self, workspace_id: str) -> tuple[dict[str, Any], Path]:
        if not self.workspace_registry:
            raise ValueError("git_workspace_not_found")
        workspace = self.workspace_registry.get(workspace_id)
        if not workspace:
            raise ValueError("git_workspace_not_found")
        if workspace.get("status") not in READY_WORKSPACE_STATUSES:
            raise ValueError("workspace_not_ready")
        raw = workspace.get("path") or ""
        if not raw:
            raise ValueError("workspace_path_missing")
        worktree = Path(raw).resolve()
        if not worktree.is_dir():
            raise ValueError("workspace_path_missing")
        source = repo_mount(workspace["repo_id"])
        if source and worktree == source.resolve():
            raise ValueError("refuses_source_checkout")
        return workspace, worktree

    def _validate_payload(self, op: str, payload: dict[str, Any], worktree: Path) -> None:
        if op == "write_file":
            path = payload.get("path")
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("write_file_content_required")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("write_file_too_large")
            safe_worktree_path(worktree, str(path) if path is not None else "")
            return
        if op == "run_tests":
            command = str(payload.get("command", "")).strip()
            if command not in TEST_COMMANDS:
                raise ValueError("test_command_not_allowlisted")
            return
        if op == "commit":
            message = str(payload.get("message", "")).strip()
            if not message:
                raise ValueError("commit_message_required")
            paths = payload.get("paths")
            if not isinstance(paths, list) or not paths:
                raise ValueError("commit_paths_required")
            for item in paths:
                if not isinstance(item, str) or item.startswith("-"):
                    raise ValueError("path_escape")
                safe_worktree_path(worktree, item)
            return
        raise ValueError("unknown_op")

    def _execute(self, op: str, payload: dict[str, Any], worktree: Path) -> dict[str, Any]:
        if op == "write_file":
            target = safe_worktree_path(worktree, str(payload["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            content = payload["content"]
            target.write_text(content, encoding="utf-8")
            return {
                "path": str(target.relative_to(worktree.resolve())),
                "bytes": len(content.encode("utf-8")),
            }
        if op == "run_tests":
            command = str(payload["command"]).strip()
            argv = TEST_COMMANDS[command]
            env = {**os.environ, "CI": "true"}
            if argv[:1] == ["pytest"]:
                env["PYTEST_ADDOPTS"] = ""
                env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            result = subprocess.run(
                argv,
                cwd=str(worktree),
                check=False,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
                env=env,
            )
            return {
                "command": command,
                "returncode": result.returncode,
                "ok": result.returncode == 0,
                "stdout": (result.stdout or "")[:MAX_TEST_OUTPUT],
                "stderr": (result.stderr or "")[:MAX_TEST_OUTPUT],
            }
        if op == "commit":
            message = str(payload["message"]).strip()
            paths = [str(item) for item in payload["paths"]]
            for relative in paths:
                safe_worktree_path(worktree, relative)
            added = _run_git(worktree, "add", "--", *paths)
            if added.returncode != 0:
                error = added.stderr.strip() or added.stdout.strip() or "git add failed"
                raise ValueError(error[:2_000])
            committed = _run_git(worktree, "commit", "-m", message, "--", *paths)
            if committed.returncode != 0:
                error = committed.stderr.strip() or committed.stdout.strip() or "git commit failed"
                raise ValueError(error[:2_000])
            sha = _run_git(worktree, "rev-parse", "HEAD")
            return {
                "sha": sha.stdout.strip() if sha.returncode == 0 else "",
                "message": message,
                "paths": paths,
            }
        raise ValueError("unknown_op")

    @staticmethod
    def _default_title(op: str, payload: dict[str, Any]) -> str:
        if op == "write_file":
            return f"Write {payload.get('path', 'file')}"
        if op == "run_tests":
            return f"Run {payload.get('command', 'tests')}"
        if op == "commit":
            message = str(payload.get("message", "commit")).strip()
            return f"Commit: {message[:72]}"
        return op

    def _refresh_workspace(self, workspace_id: str) -> None:
        if not self.workspace_registry:
            return
        try:
            self.workspace_registry.refresh(workspace_id)
        except ValueError:
            logger.warning("Could not refresh workspace %s after write", workspace_id)

    def _fail(self, write: dict[str, Any], error: str) -> dict[str, Any]:
        now = _now()
        write.update({"status": "failed", "error": error[:2_000], "updated_at": now})
        write["events"].append({"at": now, "action": "failed", "detail": error[:2_000]})
        self.save()
        return write

    def save(self) -> None:
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self.data_file.write_text(json.dumps({"writes": self.writes}, indent=2))
        except OSError as exc:
            logger.error("Could not save Git writes: %s", exc)

    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            payload = json.loads(self.data_file.read_text())
            self.writes = payload.get("writes", [])[:MAX_WRITES]
            for write in self.writes:
                write.setdefault("events", [])
                write.setdefault("result", {})
                write.setdefault("error", "")
                if write.get("status") == "approved":
                    write["status"] = "failed"
                    write["error"] = "interrupted_by_restart"
                    write["updated_at"] = _now()
            self.save()
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load Git writes: %s", exc)
