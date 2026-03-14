"""
AIIA Autonomous Task Runner — Scheduled background tasks for AIIA.

Gives AIIA autonomy: health journaling, repo sync, memory consolidation,
daily briefs. Tasks run serially to avoid resource contention on Mac Mini.

Tasks call AIIA via HTTP (port 8100) — no direct ChromaDB imports.
State persisted to task_data.json. Progress broadcast via WebSocket for
console visualization.

Usage:
    task_runner = TaskRunner(broadcast_fn=manager.broadcast, repo_path="/path/to/repo", monitor_state=monitor)
    task_runner.load_state()
    asyncio.create_task(task_runner.run_loop())
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

from local_brain.scripts.daily_report import generate_report

logger = logging.getLogger("aiia.tasks")

AIIA_BASE_URL = "http://localhost:8100"
TASK_DATA_FILE = Path(__file__).parent / "task_data.json"
MAX_RUN_HISTORY = 10
MAX_INSIGHTS = 50
SCHEDULER_INTERVAL = 10  # seconds between schedule checks

# ─────────────────────────────────────────────────────────────
# File indexing utilities (copied from bootstrap.py to avoid
# transitive ChromaDB/KnowledgeStore dependency)
# ─────────────────────────────────────────────────────────────

SKIP_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    ".venv-ml",
    "__pycache__",
    ".git",
    ".next",
    ".claude",
    "dist",
    "build",
    ".vercel",
    ".cache",
    "chroma",
    "orchestration",
    "deployments",
}

SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".egg",
    ".whl",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".lock",
    ".map",
}

INDEXABLE_EXTENSIONS = {
    ".md",
    ".py",
    ".yaml",
    ".yml",
    ".txt",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
}

MAX_FILE_SIZE = 100_000
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _should_index(filepath: str) -> bool:
    """Check if a file should be indexed."""
    path = Path(filepath)
    if any(part.startswith(".") for part in path.parts if part != "."):
        return False
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    if path.suffix in SKIP_EXTENSIONS:
        return False
    if path.name.startswith(".env"):
        return False
    if path.suffix not in INDEXABLE_EXTENSIONS:
        return False
    try:
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return True


def _chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with deterministic IDs."""
    if len(text) <= CHUNK_SIZE:
        chunk_id = hashlib.sha256(f"{source}:0".encode()).hexdigest()[:16]
        return [{"id": chunk_id, "text": text, "source": source, "chunk_index": 0}]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            para_break = text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if para_break > start:
                end = para_break + 2
            else:
                line_break = text.rfind("\n", start + CHUNK_SIZE // 2, end)
                if line_break > start:
                    end = line_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunk_id = hashlib.sha256(f"{source}:{chunk_index}".encode()).hexdigest()[
                :16
            ]
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": source,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks


def _detect_doc_type(filepath: str) -> str:
    """Detect document type from file path."""
    path = Path(filepath)
    if path.suffix == ".md":
        return "documentation"
    if path.suffix in (".yaml", ".yml"):
        return "configuration"
    if path.suffix == ".py":
        if "agents" in path.parts:
            return "agent_code"
        if "routes" in path.parts:
            return "route_code"
        if "services" in path.parts:
            return "service_code"
        if "models" in path.parts:
            return "model_code"
        if "local_brain" in path.parts:
            return "local_brain_code"
        return "code"
    return "other"


# ─────────────────────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────────────────────

TASK_DEFINITIONS = {
    "health_journal": {
        "name": "Health Journal",
        "description": "Snapshot monitor health into AIIA memory",
        "schedule_seconds": 3600,  # every hour
        "uses_llm": False,
    },
    "repo_sync": {
        "name": "Repo Sync",
        "description": "Git pull + re-index changed files into ChromaDB",
        "schedule_seconds": 21600,  # every 6 hours
        "uses_llm": False,
    },
    "code_health": {
        "name": "Code Health",
        "description": "Lint, syntax check, track TODOs and code quality trends",
        "schedule_seconds": 10800,  # every 3 hours
        "uses_llm": False,
    },
    "learning_loop": {
        "name": "Learning Loop",
        "description": "Review recent code changes, extract patterns and decisions",
        "schedule_seconds": 14400,  # every 4 hours
        "uses_llm": True,
    },
    "test_runner": {
        "name": "Test Runner",
        "description": "Run test suite, track pass/fail trends",
        "schedule_seconds": 14400,  # every 4 hours
        "uses_llm": False,
    },
    "memory_digest": {
        "name": "Memory Digest",
        "description": "AIIA reviews her own memories for duplicates/conflicts",
        "schedule_cron_hour": 6,
        "schedule_cron_minute": 0,
        "uses_llm": True,
    },
    "daily_brief": {
        "name": "Daily Brief",
        "description": "Generate summary of git activity, health trends, KB growth",
        "schedule_cron_hour": 8,
        "schedule_cron_minute": 0,
        "uses_llm": True,
    },
    "weekly_client_status": {
        "name": "Client Weekly Status",
        "description": "Generate client delivery report: features shipped, bugs fixed, metrics",
        "schedule_seconds": 604800,  # 7 days
        "uses_llm": True,
    },
    "cross_tenant_analytics": {
        "name": "Cross-Tenant Analytics",
        "description": "Aggregate conversation patterns across tenants (read-only DB)",
        "schedule_cron_hour": 3,
        "schedule_cron_minute": 0,
        "uses_llm": True,
    },
    "security_scan": {
        "name": "Security Scan",
        "description": "Check Dependabot alerts, CI status, dependency vulnerabilities",
        "schedule_seconds": 21600,  # every 6 hours
        "uses_llm": False,
    },
    "ci_monitor": {
        "name": "CI Monitor",
        "description": "Poll GitHub Actions for failures, create action items",
        "schedule_seconds": 1800,  # every 30 minutes
        "uses_llm": False,
    },
}


# ─────────────────────────────────────────────────────────────
# TaskRunner
# ─────────────────────────────────────────────────────────────


class TaskRunner:
    """Manages scheduling, execution, progress, and persistence for AIIA tasks."""

    def __init__(
        self,
        broadcast_fn: Callable[[str, Any], Coroutine],
        repo_path: str,
        monitor_state: Any,
        action_queue: Any = None,
    ):
        self.broadcast = broadcast_fn
        self.repo_path = repo_path
        self.monitor_state = monitor_state
        self.action_queue = action_queue
        self._running_task: Optional[str] = None

        # Initialize task states
        self.tasks: Dict[str, Dict[str, Any]] = {}
        for task_id, defn in TASK_DEFINITIONS.items():
            self.tasks[task_id] = {
                "task_id": task_id,
                "name": defn["name"],
                "description": defn["description"],
                "uses_llm": defn.get("uses_llm", False),
                "status": "idle",
                "progress": 0,
                "current_step": "",
                "last_run": None,
                "last_result": None,
                "last_output": None,  # Full readable output (briefs, reports)
                "last_duration_ms": None,
                "next_run": None,
                "run_count": 0,
                "fail_count": 0,
                "run_history": [],
            }

        # Extra persistent state (e.g. last_commit_sha for repo_sync)
        self._extra: Dict[str, Any] = {}

        # Task implementations
        self._implementations = {
            "health_journal": self._task_health_journal,
            "repo_sync": self._task_repo_sync,
            "code_health": self._task_code_health,
            "learning_loop": self._task_learning_loop,
            "test_runner": self._task_test_runner,
            "memory_digest": self._task_memory_digest,
            "daily_brief": self._task_daily_brief,
            "weekly_client_status": self._task_weekly_client_status,
            "cross_tenant_analytics": self._task_cross_tenant_analytics,
            "security_scan": self._task_security_scan,
            "ci_monitor": self._task_ci_monitor,
        }

    # ─── Scheduling ──────────────────────────────────────────

    async def run_loop(self):
        """Main scheduler loop — checks every SCHEDULER_INTERVAL seconds."""
        await asyncio.sleep(5)  # let server finish startup
        logger.info(f"Task Runner started — {len(TASK_DEFINITIONS)} tasks registered")

        # Calculate initial next_run for all tasks
        for task_id in self.tasks:
            self._update_next_run(task_id)

        while True:
            try:
                if self._running_task is None:
                    due_task = self._find_due_task()
                    if due_task:
                        await self._execute_task(due_task)
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")

            await asyncio.sleep(SCHEDULER_INTERVAL)

    def _find_due_task(self) -> Optional[str]:
        """Find the next task that is due to run."""
        now = datetime.now(timezone.utc)

        for task_id, defn in TASK_DEFINITIONS.items():
            task = self.tasks[task_id]
            if task["status"] == "running":
                continue

            last_run = task["last_run"]

            # Interval-based schedule
            if "schedule_seconds" in defn:
                if last_run is None:
                    return task_id
                last_dt = datetime.fromisoformat(last_run)
                elapsed = (now - last_dt).total_seconds()
                if elapsed >= defn["schedule_seconds"]:
                    return task_id

            # Cron-based schedule
            if "schedule_cron_hour" in defn:
                if now.hour == defn["schedule_cron_hour"] and now.minute == defn.get(
                    "schedule_cron_minute", 0
                ):
                    # Only if we haven't run in the last 23 hours
                    if last_run is None:
                        return task_id
                    last_dt = datetime.fromisoformat(last_run)
                    elapsed_hours = (now - last_dt).total_seconds() / 3600
                    if elapsed_hours >= 23:
                        return task_id

        return None

    def _update_next_run(self, task_id: str):
        """Calculate and store next_run for a task."""
        defn = TASK_DEFINITIONS[task_id]
        task = self.tasks[task_id]
        now = datetime.now(timezone.utc)

        if "schedule_seconds" in defn:
            if task["last_run"]:
                last_dt = datetime.fromisoformat(task["last_run"])
                next_dt = (
                    last_dt.replace(tzinfo=timezone.utc)
                    if last_dt.tzinfo is None
                    else last_dt
                )
                from datetime import timedelta

                next_dt = next_dt + timedelta(seconds=defn["schedule_seconds"])
            else:
                next_dt = now  # run immediately on first startup
            task["next_run"] = next_dt.isoformat()

        elif "schedule_cron_hour" in defn:
            from datetime import timedelta

            target_hour = defn["schedule_cron_hour"]
            target_minute = defn.get("schedule_cron_minute", 0)
            next_dt = now.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0
            )
            if next_dt <= now:
                next_dt += timedelta(days=1)
            task["next_run"] = next_dt.isoformat()

    # ─── Execution Lifecycle ─────────────────────────────────

    async def trigger_task(self, task_id: str) -> Dict[str, Any]:
        """Manually trigger a task. Returns immediately; task runs in background."""
        if task_id not in self.tasks:
            return {"error": f"Unknown task: {task_id}"}
        if self._running_task is not None:
            return {
                "error": f"Task '{self._running_task}' is already running",
                "queued": False,
            }

        asyncio.create_task(self._execute_task(task_id))
        return {"status": "triggered", "task_id": task_id}

    async def _execute_task(self, task_id: str):
        """Full lifecycle: start -> progress -> complete/fail -> persist."""
        task = self.tasks[task_id]
        self._running_task = task_id
        start_time = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        # Mark as running
        task["status"] = "running"
        task["progress"] = 0
        task["current_step"] = "Starting..."

        await self.broadcast(
            "task_started",
            {
                "task_id": task_id,
                "name": task["name"],
                "timestamp": started_at,
            },
        )
        await self._broadcast_task_update()

        try:
            impl = self._implementations[task_id]
            result = await impl()

            # Tasks can return (summary, full_output) or just summary
            if isinstance(result, tuple):
                result_summary, full_output = result
            else:
                result_summary = result
                full_output = None

            elapsed_ms = round((time.monotonic() - start_time) * 1000, 1)
            finished_at = datetime.now(timezone.utc).isoformat()

            # Mark success
            task["status"] = "done"
            task["progress"] = 100
            task["current_step"] = "Complete"
            task["last_run"] = finished_at
            task["last_result"] = result_summary
            task["last_output"] = full_output
            task["last_duration_ms"] = elapsed_ms
            task["run_count"] += 1

            # Add to history
            task["run_history"].insert(
                0,
                {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "done",
                    "duration_ms": elapsed_ms,
                    "summary": result_summary,
                },
            )
            task["run_history"] = task["run_history"][:MAX_RUN_HISTORY]

            await self.broadcast(
                "task_complete",
                {
                    "task_id": task_id,
                    "result_summary": result_summary,
                    "elapsed_ms": elapsed_ms,
                },
            )
            logger.info(
                f"Task '{task_id}' completed in {elapsed_ms}ms: {result_summary}"
            )

        except Exception as e:
            elapsed_ms = round((time.monotonic() - start_time) * 1000, 1)
            finished_at = datetime.now(timezone.utc).isoformat()
            error_msg = str(e)[:200]

            task["status"] = "failed"
            task["progress"] = 0
            task["current_step"] = f"Error: {error_msg}"
            task["last_run"] = finished_at
            task["last_result"] = f"FAILED: {error_msg}"
            task["last_duration_ms"] = elapsed_ms
            task["fail_count"] += 1

            task["run_history"].insert(
                0,
                {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "failed",
                    "duration_ms": elapsed_ms,
                    "summary": f"FAILED: {error_msg}",
                },
            )
            task["run_history"] = task["run_history"][:MAX_RUN_HISTORY]

            await self.broadcast(
                "task_failed",
                {
                    "task_id": task_id,
                    "error": error_msg,
                    "elapsed_ms": elapsed_ms,
                },
            )
            logger.error(f"Task '{task_id}' failed after {elapsed_ms}ms: {error_msg}")

        finally:
            self._running_task = None
            self._update_next_run(task_id)
            await self._broadcast_task_update()
            self.save_state()

    async def _progress(self, task_id: str, pct: int, step: str, message: str = ""):
        """Broadcast a progress update for a running task."""
        task = self.tasks[task_id]
        task["progress"] = pct
        task["current_step"] = step

        await self.broadcast(
            "task_progress",
            {
                "task_id": task_id,
                "progress_pct": pct,
                "step": step,
                "message": message,
            },
        )

    async def _broadcast_task_update(self):
        """Broadcast full snapshot of all tasks."""
        await self.broadcast("task_update", self.get_all_tasks())

    # ─── API Methods ─────────────────────────────────────────

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Return all tasks with current status."""
        return list(self.tasks.values())

    def get_history(self) -> List[Dict[str, Any]]:
        """Return recent run history across all tasks."""
        all_runs = []
        for task_id, task in self.tasks.items():
            for run in task["run_history"]:
                all_runs.append({"task_id": task_id, "name": task["name"], **run})
        all_runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return all_runs[:30]

    # ─── Persistence ─────────────────────────────────────────

    def save_state(self):
        """Persist task state to JSON."""
        try:
            data = {
                "tasks": {},
                "extra": self._extra,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            for task_id, task in self.tasks.items():
                # Cap output at 3000 chars for persistence
                output = task.get("last_output")
                if output and len(output) > 3000:
                    output = output[:2997] + "..."
                data["tasks"][task_id] = {
                    "last_run": task["last_run"],
                    "last_result": task["last_result"],
                    "last_output": output,
                    "last_duration_ms": task["last_duration_ms"],
                    "run_count": task["run_count"],
                    "fail_count": task["fail_count"],
                    "run_history": task["run_history"],
                }
            TASK_DATA_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to persist task data: {e}")

    def load_state(self):
        """Load persisted task state from JSON."""
        if not TASK_DATA_FILE.exists():
            logger.info("No task data file found — starting fresh")
            return

        try:
            data = json.loads(TASK_DATA_FILE.read_text())
            self._extra = data.get("extra", {})

            for task_id, saved in data.get("tasks", {}).items():
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task["last_run"] = saved.get("last_run")
                    task["last_result"] = saved.get("last_result")
                    task["last_output"] = saved.get("last_output")
                    task["last_duration_ms"] = saved.get("last_duration_ms")
                    task["run_count"] = saved.get("run_count", 0)
                    task["fail_count"] = saved.get("fail_count", 0)
                    task["run_history"] = saved.get("run_history", [])

            total_runs = sum(t["run_count"] for t in self.tasks.values())
            logger.info(
                f"Loaded task state: {total_runs} total runs across {len(self.tasks)} tasks"
            )
        except Exception as e:
            logger.warning(f"Could not load task data: {e}")

    # ─── Shared Helpers ──────────────────────────────────────

    async def _aiia_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """HTTP request to AIIA at port 8100."""
        url = f"{AIIA_BASE_URL}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url)
            elif method == "POST":
                resp = await client.post(url, json=body or {})
            else:
                raise ValueError(f"Unsupported method: {method}")

            if resp.status_code != 200:
                raise RuntimeError(
                    f"AIIA returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return resp.json()

    async def _run_git(self, *args: str) -> str:
        """Run a git command asynchronously. Returns stdout."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error = stderr.decode().strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {error}")
        return stdout.decode().strip()

    # ─── Insight Emission ─────────────────────────────────────

    async def _emit_insight(
        self,
        insight_type: str,
        severity: str,
        title: str,
        detail: str = "",
        source_task: str = "",
    ):
        """Emit a structured insight into _extra["insights"] and broadcast via WebSocket.

        Args:
            insight_type: "shipped", "security", "quality", "memory", "metric", "learning"
            severity: "info", "warn", "error", "success"
            title: Short headline
            detail: Extended description
            source_task: Which task generated this
        """
        insight = {
            "type": insight_type,
            "severity": severity,
            "title": title,
            "detail": detail,
            "source_task": source_task,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        insights = self._extra.setdefault("insights", [])
        insights.insert(0, insight)
        # Cap at MAX_INSIGHTS
        self._extra["insights"] = insights[:MAX_INSIGHTS]

        # Broadcast to connected dashboards
        await self.broadcast("new_insight", insight)

    async def _run_command(self, *args: str, timeout: float = 30.0) -> str:
        """Run an arbitrary command asynchronously. Returns stdout."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(args)}")
        if proc.returncode != 0:
            error = stderr.decode().strip()
            raise RuntimeError(f"{' '.join(args)} failed: {error}")
        return stdout.decode().strip()

    # ─── Task Implementations ────────────────────────────────

    async def _task_health_journal(self) -> str:
        """Snapshot monitor health into AIIA memory."""
        await self._progress("health_journal", 10, "Reading monitor state")

        snapshot = self.monitor_state.get_full_snapshot()
        services = snapshot.get("services", {})

        await self._progress("health_journal", 30, "Formatting health snapshot")

        # Build a compact health string
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        parts = []
        for sid, svc in services.items():
            status = svc.get("status", "unknown")
            rt = svc.get("response_time_ms")
            rt_str = f" ({rt}ms)" if rt is not None else ""
            parts.append(f"{svc.get('name', sid)} {status}{rt_str}")

        health_text = f"Health snapshot {now}: " + ", ".join(parts)

        await self._progress("health_journal", 60, "Storing in AIIA memory")

        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": health_text,
                "category": "project",
                "source": "task:health_journal",
            },
        )

        # Emit insight if any service is degraded/offline
        offline = [
            svc.get("name", sid)
            for sid, svc in services.items()
            if svc.get("status") != "online"
        ]
        if offline:
            await self._emit_insight(
                "quality",
                "warn",
                f"{len(offline)} service(s) not healthy: {', '.join(offline)}",
                source_task="health_journal",
            )
        else:
            await self._emit_insight(
                "metric",
                "success",
                f"All {len(services)} services healthy",
                source_task="health_journal",
            )

        await self._progress("health_journal", 100, "Complete")
        return f"Stored health snapshot: {len(services)} services checked"

    async def _task_repo_sync(self) -> str:
        """Git pull + re-index changed files into ChromaDB."""
        await self._progress("repo_sync", 5, "Getting current commit SHA")

        current_sha = await self._run_git("rev-parse", "HEAD")
        last_sha = self._extra.get("last_commit_sha")

        await self._progress("repo_sync", 10, "Pulling from origin")

        try:
            pull_result = await self._run_git("pull", "--ff-only", "origin", "main")
        except RuntimeError as e:
            # Non-ff pull or network error — still continue with what we have
            pull_result = f"Pull failed (non-fatal): {e}"
            logger.warning(pull_result)

        await self._progress("repo_sync", 20, "Detecting changed files")

        new_sha = await self._run_git("rev-parse", "HEAD")

        if last_sha and last_sha != new_sha:
            # Get changed files between old and new
            diff_output = await self._run_git(
                "diff", "--name-only", f"{last_sha}..{new_sha}"
            )
            changed_files = [f for f in diff_output.strip().split("\n") if f]
        elif last_sha is None:
            # First run — don't index everything, just mark position
            self._extra["last_commit_sha"] = new_sha
            await self._progress("repo_sync", 100, "Complete")
            return f"First run — stored baseline SHA {new_sha[:8]}. No files indexed."
        else:
            # No changes
            self._extra["last_commit_sha"] = new_sha
            await self._progress("repo_sync", 100, "Complete")
            return "Already up to date — no new commits"

        await self._progress(
            "repo_sync", 30, f"Filtering {len(changed_files)} changed files"
        )

        # Filter to indexable files
        indexable = []
        for rel_path in changed_files:
            full_path = os.path.join(self.repo_path, rel_path)
            if os.path.exists(full_path) and _should_index(full_path):
                indexable.append(rel_path)

        if not indexable:
            self._extra["last_commit_sha"] = new_sha
            await self._progress("repo_sync", 100, "Complete")
            commits = (
                new_sha[:8] if last_sha is None else f"{last_sha[:8]}..{new_sha[:8]}"
            )
            return (
                f"Pulled {commits}: {len(changed_files)} changed files, none indexable"
            )

        await self._progress("repo_sync", 40, f"Indexing {len(indexable)} files")

        # Index each file into AIIA via ingest endpoint
        indexed_count = 0
        total_chunks = 0
        for i, rel_path in enumerate(indexable):
            pct = 40 + int((i / len(indexable)) * 50)  # 40-90%
            await self._progress("repo_sync", pct, f"Indexing {rel_path}")

            full_path = os.path.join(self.repo_path, rel_path)
            try:
                text = Path(full_path).read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue

                doc_type = _detect_doc_type(rel_path)
                chunks = _chunk_text(text, rel_path)

                for chunk in chunks:
                    await self._aiia_request(
                        "POST",
                        "/v1/aiia/ingest",
                        {
                            "text": chunk["text"],
                            "source": rel_path,
                            "doc_type": doc_type,
                            "metadata": {
                                "chunk_index": chunk["chunk_index"],
                                "synced_from": new_sha[:8],
                            },
                        },
                    )
                    total_chunks += 1

                indexed_count += 1
            except Exception as e:
                logger.warning(f"Failed to index {rel_path}: {e}")

        self._extra["last_commit_sha"] = new_sha
        await self._progress("repo_sync", 100, "Complete")

        commits_str = f"{last_sha[:8]}..{new_sha[:8]}" if last_sha else new_sha[:8]

        if indexed_count > 0:
            await self._emit_insight(
                "shipped",
                "info",
                f"Synced {indexed_count} files ({total_chunks} chunks) from {commits_str}",
                source_task="repo_sync",
            )

        return f"Pulled {commits_str}: re-indexed {indexed_count} files ({total_chunks} chunks)"

    async def _task_memory_digest(self) -> str:
        """AIIA reviews her own memories for duplicates and conflicts."""
        await self._progress("memory_digest", 5, "Loading memories from AIIA")

        memories_data = await self._aiia_request("GET", "/v1/aiia/memory?limit=200")
        memories = memories_data.get("memories", [])

        if not memories:
            await self._progress("memory_digest", 100, "Complete")
            return "No memories to review"

        await self._progress(
            "memory_digest", 15, f"Grouping {len(memories)} memories by category"
        )

        # Group by category
        by_category: Dict[str, List] = {}
        for mem in memories:
            cat = mem.get("category", "uncategorized")
            by_category.setdefault(cat, []).append(mem)

        await self._progress(
            "memory_digest", 20, f"Found {len(by_category)} categories"
        )

        # Review categories with enough entries to have potential duplicates
        findings = []
        categories_reviewed = 0
        reviewable = {cat: mems for cat, mems in by_category.items() if len(mems) > 3}

        for i, (cat, mems) in enumerate(reviewable.items()):
            pct = 20 + int((i / max(len(reviewable), 1)) * 50)  # 20-70%
            await self._progress(
                "memory_digest", pct, f"Reviewing '{cat}' ({len(mems)} entries)"
            )

            # Build prompt with memory contents
            memory_list = "\n".join(
                f"  [{j + 1}] {m.get('fact', '')[:200]}" for j, m in enumerate(mems)
            )

            prompt = (
                f"Review these {len(mems)} memories in the '{cat}' category. "
                f"Identify any duplicates, contradictions, or outdated entries. "
                f"Respond with a brief JSON object: "
                f'{{"duplicates": ["description..."], "contradictions": ["description..."], "outdated": ["description..."], "clean": true/false}}\n\n'
                f"Memories:\n{memory_list}"
            )

            try:
                result = await self._aiia_request(
                    "POST",
                    "/v1/aiia/ask",
                    {
                        "question": prompt,
                        "context": f"You are reviewing your own memories for quality. Category: {cat}",
                        "n_results": 0,
                    },
                    timeout=60.0,
                )

                answer = result.get("answer", "")
                findings.append(f"[{cat}] {answer[:300]}")
                categories_reviewed += 1
            except Exception as e:
                logger.warning(f"Memory digest review failed for '{cat}': {e}")
                findings.append(f"[{cat}] Review failed: {str(e)[:100]}")

        await self._progress("memory_digest", 75, "Storing digest findings")

        # Store findings as a memory
        if findings:
            digest_text = (
                f"Memory Digest {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: "
                f"Reviewed {categories_reviewed} categories ({len(memories)} total memories). "
                + " | ".join(findings)
            )
            # Truncate if too long
            if len(digest_text) > 2000:
                digest_text = digest_text[:1997] + "..."

            await self._aiia_request(
                "POST",
                "/v1/aiia/remember",
                {
                    "fact": digest_text,
                    "category": "meta",
                    "source": "task:memory_digest",
                },
            )

        await self._emit_insight(
            "memory",
            "info",
            f"Memory digest: reviewed {categories_reviewed} categories ({len(memories)} memories)",
            detail="; ".join(findings[:3]) if findings else "",
            source_task="memory_digest",
        )

        await self._progress("memory_digest", 100, "Complete")
        return (
            f"Reviewed {categories_reviewed} categories, {len(memories)} memories total"
        )

    async def _task_daily_brief(self) -> str:
        """Generate a daily summary: git activity, health trends, KB growth."""
        await self._progress("daily_brief", 5, "Gathering git activity")

        # Per-product commit report via daily_report.py
        report = None
        recent_commits = []
        try:
            report = generate_report(repo_dir=self.repo_path)
            # Flatten commit list for counting
            for product_data in report.get("products", {}).values():
                recent_commits.extend(product_data.get("commits", []))
        except Exception as e:
            logger.warning("generate_report() failed, falling back to git log: %s", e)

        # Fallback: raw git log if report generation failed
        if report is None:
            try:
                git_log = await self._run_git(
                    "log", "--oneline", "--since=24 hours ago"
                )
                recent_commits = [line for line in git_log.strip().split("\n") if line]
            except Exception:
                recent_commits = []

        await self._progress("daily_brief", 15, "Reading health trends")

        # Health snapshot from monitor
        snapshot = self.monitor_state.get_full_snapshot()
        services = snapshot.get("services", {})
        health_lines = []
        for sid, svc in services.items():
            status = svc.get("status", "unknown")
            uptime = svc.get("uptime_pct")
            uptime_str = f", {uptime}% uptime" if uptime is not None else ""
            avg_rt = svc.get("avg_response_time_ms")
            rt_str = f", avg {avg_rt}ms" if avg_rt else ""
            health_lines.append(f"{svc.get('name', sid)}: {status}{uptime_str}{rt_str}")

        await self._progress("daily_brief", 25, "Checking AIIA knowledge stats")

        # KB stats
        try:
            aiia_status = await self._aiia_request(
                "GET", "/v1/aiia/status", timeout=10.0
            )
            knowledge = aiia_status.get("knowledge", {})
            kb_docs = knowledge.get("knowledge_docs", 0)
            memory_info = aiia_status.get("memory", {})
            total_memories = memory_info.get("total_memories", 0)
            model = aiia_status.get("model", "unknown")
        except Exception:
            kb_docs = 0
            total_memories = 0
            model = "unreachable"

        # Compare with yesterday's stats
        yesterday_stats = self._extra.get("daily_brief_stats", {})
        kb_delta = kb_docs - yesterday_stats.get("kb_docs", kb_docs)
        mem_delta = total_memories - yesterday_stats.get(
            "total_memories", total_memories
        )

        await self._progress("daily_brief", 35, "Loading recent memories")

        # Recent memories
        try:
            mem_data = await self._aiia_request(
                "GET", "/v1/aiia/memory?limit=20", timeout=10.0
            )
            recent_memories = mem_data.get("memories", [])
        except Exception:
            recent_memories = []

        await self._progress("daily_brief", 40, "Checking token usage")

        # Token usage from Command Center
        token_section = "  No token data available"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:8200/api/tokens/today")
                if resp.status_code == 200:
                    tdata = resp.json()
                    total_tokens = tdata.get("total_tokens", 0)
                    total_cost = tdata.get("total_cost", 0)
                    total_requests = tdata.get("total_requests", 0)
                    providers = tdata.get("by_provider", {})
                    parts = []
                    for prov, stats in providers.items():
                        cost = (
                            "FREE"
                            if prov == "local"
                            else f"${stats.get('cost', 0):.4f}"
                        )
                        parts.append(
                            f"{prov}: {stats.get('tokens', 0)} tokens ({cost})"
                        )
                    token_section = (
                        f"  Total: {total_tokens} tokens, ${total_cost:.4f}, {total_requests} requests\n  "
                        + "\n  ".join(parts)
                    )
        except Exception:
            pass

        await self._progress("daily_brief", 43, "Checking WIP items")

        # WIP items
        wip_section = "  No WIP items"
        try:
            wip_data = await self._aiia_request(
                "GET", "/v1/aiia/memory?category=wip&limit=10", timeout=10.0
            )
            wip_items = wip_data.get("memories", [])
            if wip_items:
                wip_section = "\n".join(
                    f"  - {w.get('fact', '')[:120]}" for w in wip_items[:5]
                )
        except Exception:
            pass

        await self._progress("daily_brief", 43, "Loading project priorities")

        # Project priorities from AIIA memories
        priorities_section = "  No priority data"
        try:
            pri_data = await self._aiia_request(
                "GET", "/v1/aiia/memory?category=project&limit=20", timeout=10.0
            )
            pri_items = pri_data.get("memories", [])
            # Filter for priority/deadline/workstream keywords
            priority_keywords = {
                "priority",
                "workstream",
                "deadline",
                "client",
                "revenue",
                "launch",
                "milestone",
                "blocker",
            }
            relevant = [
                m
                for m in pri_items
                if any(kw in m.get("fact", "").lower() for kw in priority_keywords)
            ]
            if relevant:
                priorities_section = "\n".join(
                    f"  - {m.get('fact', '')[:150]}" for m in relevant[:8]
                )
        except Exception:
            pass

        await self._progress("daily_brief", 45, "Generating brief with AIIA")

        # Build per-product commits section from report data
        if report and report.get("products"):
            summary = report["summary"]
            commits_lines = [
                f"  Total: {summary['total_commits']} commits, "
                f"{summary['total_files_changed']} files, "
                f"+{summary['total_additions']}/-{summary['total_deletions']}"
            ]
            for product, pdata in report["products"].items():
                commits_lines.append(
                    f"\n  [{product}] — {pdata['commit_count']} commits, "
                    f"+{pdata['total_additions']}/-{pdata['total_deletions']}"
                )
                for c in pdata["commits"][:5]:
                    commits_lines.append(
                        f"    {c['type']:8s} {c['hash']} {c['subject'][:70]}"
                    )
            # Syntax errors from report
            sx = report.get("syntax_errors", {})
            if sx.get("total_errors", 0) > 0:
                commits_lines.append(
                    f"\n  SYNTAX ERRORS: {sx['total_errors']} errors found"
                )
            commits_section = "\n".join(commits_lines)
        elif recent_commits:
            # Fallback: flat list from raw git log
            commits_section = "\n".join(f"  - {c}" for c in recent_commits[:15])
        else:
            commits_section = "  No commits in last 24h"

        health_section = "\n".join(f"  - {h}" for h in health_lines)
        memory_section = "\n".join(
            f"  - [{m.get('category', '?')}] {m.get('fact', '')[:100]}"
            for m in recent_memories[:10]
        )

        prompt = f"""Generate a morning briefing for the development team lead. Structure your response in exactly 4 sections:

## 1. Shipped Code (by product)
Summarize what was built yesterday, grouped by product. Highlight notable features, fixes, and patterns.

## 2. Priority Recommendations
Based on WIP items, project priorities, and what shipped — what should the team focus on today? Consider: client deadlines, revenue impact, security items, unfinished work.

## 3. Platform Health
Brief service status — what's up, what's degraded, any concerns.

## 4. Key Numbers
Token usage, KB growth, one-liners. Keep this tight.

Data:

SHIPPED CODE (by product, last 24h):
{commits_section}

PLATFORM HEALTH:
{health_section}

TOKEN USAGE:
{token_section}

KB STATS: {kb_docs} docs ({"+" if kb_delta >= 0 else ""}{kb_delta}), {total_memories} memories ({"+" if mem_delta >= 0 else ""}{mem_delta}), model: {model}

WORK IN PROGRESS:
{wip_section}

PROJECT PRIORITIES:
{priorities_section}

RECENT MEMORIES:
{memory_section}"""

        try:
            result = await self._aiia_request(
                "POST",
                "/v1/aiia/ask",
                {
                    "question": prompt,
                    "context": "You are AIIA generating the morning briefing for the development team. Be concise, actionable, and specific. Use the 4-section format requested. Prioritize recommendations based on revenue impact and client deadlines.",
                    "n_results": 3,
                },
                timeout=90.0,
            )
            brief = result.get(
                "answer", "Brief generation failed — no response from AIIA"
            )
        except Exception as e:
            brief = f"Brief generation failed: {str(e)[:200]}"

        await self._progress("daily_brief", 80, "Storing brief")

        # Store as memory
        brief_summary = brief[:500] if len(brief) > 500 else brief
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Daily Brief {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: {brief_summary}",
                "category": "project",
                "source": "task:daily_brief",
            },
        )

        # End session
        try:
            session_id = f"daily-brief-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            await self._aiia_request(
                "POST",
                "/v1/aiia/session-end",
                {
                    "session_id": session_id,
                    "summary": f"Daily Brief: {len(recent_commits)} commits, {kb_docs} KB docs, all services reporting",
                },
            )
        except Exception:
            pass  # session-end is optional

        await self._emit_insight(
            "metric",
            "info",
            f"Daily brief: {len(recent_commits)} commits, {kb_docs} KB docs, {total_memories} memories",
            source_task="daily_brief",
        )

        await self._progress("daily_brief", 95, "Saving stats for tomorrow")

        # Store stats for delta comparison tomorrow
        self._extra["daily_brief_stats"] = {
            "kb_docs": kb_docs,
            "total_memories": total_memories,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        await self._progress("daily_brief", 100, "Complete")
        summary = f"Brief generated: {len(recent_commits)} commits, {kb_docs} KB docs, {total_memories} memories"
        return (summary, brief)

    # ─── Continuous Tasks ────────────────────────────────────

    async def _task_code_health(self) -> str:
        """Lint, syntax check, track TODOs and code quality trends."""
        await self._progress("code_health", 5, "Scanning Python files")

        repo = Path(self.repo_path)
        py_files = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        await self._progress(
            "code_health", 15, f"Checking {len(py_files)} Python files"
        )

        # Syntax check
        syntax_errors = []
        for i, filepath in enumerate(py_files):
            if i % 50 == 0:
                pct = 15 + int((i / max(len(py_files), 1)) * 40)
                await self._progress(
                    "code_health", pct, f"Syntax checking ({i}/{len(py_files)})"
                )
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3",
                    "-m",
                    "py_compile",
                    filepath,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    rel = os.path.relpath(filepath, self.repo_path)
                    syntax_errors.append(f"{rel}: {stderr.decode().strip()[:100]}")
            except Exception:
                pass

        await self._progress("code_health", 60, "Counting TODOs and FIXMEs")

        # Count TODOs, FIXMEs, HACKs
        todo_count = 0
        fixme_count = 0
        hack_count = 0
        bare_excepts = 0
        todo_samples = []

        for filepath in py_files:
            try:
                text = Path(filepath).read_text(encoding="utf-8", errors="replace")
                rel = os.path.relpath(filepath, self.repo_path)
                for line_num, line in enumerate(text.split("\n"), 1):
                    if "TODO" in line:
                        todo_count += 1
                        if len(todo_samples) < 5:
                            todo_samples.append(f"{rel}:{line_num} {line.strip()[:80]}")
                    if "FIXME" in line:
                        fixme_count += 1
                    if "HACK" in line:
                        hack_count += 1
                    if "except:" in line and "except:  #" not in line:
                        bare_excepts += 1
            except Exception:
                pass

        await self._progress("code_health", 85, "Building report")

        # Build report
        lines = [
            f"Code Health Scan {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}",
            f"Files scanned: {len(py_files)}",
            f"Syntax errors: {len(syntax_errors)}",
            f"TODOs: {todo_count} | FIXMEs: {fixme_count} | HACKs: {hack_count}",
            f"Bare excepts: {bare_excepts}",
        ]
        if syntax_errors:
            lines.append("")
            lines.append("SYNTAX ERRORS:")
            lines.extend(f"  {e}" for e in syntax_errors[:10])
        if todo_samples:
            lines.append("")
            lines.append("SAMPLE TODOs:")
            lines.extend(f"  {t}" for t in todo_samples)

        report = "\n".join(lines)

        # Track trends in _extra
        trends = self._extra.setdefault("code_health_trends", [])
        trends.append(
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "files": len(py_files),
                "syntax_errors": len(syntax_errors),
                "todos": todo_count,
                "fixmes": fixme_count,
                "bare_excepts": bare_excepts,
            }
        )
        # Keep last 20 data points
        self._extra["code_health_trends"] = trends[-20:]

        # Emit insights on quality changes
        if len(trends) >= 2:
            prev = trends[-2]
            todo_delta = todo_count - prev.get("todos", todo_count)
            if todo_delta > 3:
                await self._emit_insight(
                    "quality",
                    "warn",
                    f"TODOs increased by {todo_delta} (now {todo_count})",
                    source_task="code_health",
                )
            err_delta = len(syntax_errors) - prev.get("syntax_errors", 0)
            if err_delta > 0:
                await self._emit_insight(
                    "quality",
                    "error",
                    f"{len(syntax_errors)} syntax errors (+{err_delta} since last scan)",
                    source_task="code_health",
                )
        if len(syntax_errors) == 0:
            await self._emit_insight(
                "quality",
                "success",
                f"Code health: {len(py_files)} files, 0 syntax errors, {todo_count} TODOs",
                source_task="code_health",
            )

        # Create action items for syntax errors
        if self.action_queue and syntax_errors:
            for err in syntax_errors[:5]:
                # Parse "relative/path.py: error message"
                parts = err.split(":", 1)
                filepath = parts[0].strip() if parts else err
                error_detail = parts[1].strip() if len(parts) > 1 else err
                self.action_queue.create_action(
                    action_type="lint_fix",
                    severity="error",
                    title=f"SyntaxError in {filepath}",
                    description=error_detail,
                    proposed_fix=f"Fix the syntax error in {filepath}",
                    source_task="code_health",
                    files_affected=[filepath],
                )
            await self.broadcast(
                "new_action",
                {"count": len(syntax_errors), "source": "code_health"},
            )

        # Store summary in AIIA
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Code health {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: {len(py_files)} files, {len(syntax_errors)} syntax errors, {todo_count} TODOs, {fixme_count} FIXMEs, {bare_excepts} bare excepts",
                "category": "project",
                "source": "task:code_health",
            },
        )

        await self._progress("code_health", 100, "Complete")
        summary = (
            f"{len(py_files)} files, {len(syntax_errors)} errors, {todo_count} TODOs"
        )
        return (summary, report)

    async def _task_learning_loop(self) -> str:
        """Review recent code changes, extract patterns and decisions into AIIA memory."""
        await self._progress("learning_loop", 5, "Getting recent git activity")

        # Get commits since last learning run
        last_learned_sha = self._extra.get("last_learned_sha")
        current_sha = await self._run_git("rev-parse", "HEAD")

        if last_learned_sha == current_sha:
            await self._progress("learning_loop", 100, "Complete")
            return "No new commits to learn from"

        await self._progress("learning_loop", 15, "Reading recent diffs")

        # Get diff — either from last learned SHA or last 12 hours
        try:
            if last_learned_sha:
                # Check if SHA still exists (might have been force-pushed)
                try:
                    await self._run_git("cat-file", "-t", last_learned_sha)
                    diff = await self._run_git(
                        "diff", "--stat", f"{last_learned_sha}..HEAD"
                    )
                    log = await self._run_git(
                        "log", "--oneline", f"{last_learned_sha}..HEAD"
                    )
                    diff_detail = await self._run_git(
                        "diff",
                        f"{last_learned_sha}..HEAD",
                        "--",
                        "*.py",
                        "*.yaml",
                        "*.md",
                    )
                except RuntimeError:
                    diff = await self._run_git("diff", "--stat", "HEAD~5..HEAD")
                    log = await self._run_git("log", "--oneline", "-5")
                    diff_detail = await self._run_git(
                        "diff", "HEAD~5..HEAD", "--", "*.py", "*.yaml", "*.md"
                    )
            else:
                diff = await self._run_git("diff", "--stat", "HEAD~5..HEAD")
                log = await self._run_git("log", "--oneline", "-5")
                diff_detail = await self._run_git(
                    "diff", "HEAD~5..HEAD", "--", "*.py", "*.yaml", "*.md"
                )
        except Exception:
            await self._progress("learning_loop", 100, "Complete")
            return "Could not read git history"

        commits = [line for line in log.strip().split("\n") if line]
        if not commits:
            self._extra["last_learned_sha"] = current_sha
            await self._progress("learning_loop", 100, "Complete")
            return "No commits to learn from"

        await self._progress("learning_loop", 30, f"Analyzing {len(commits)} commits")

        # Truncate diff to avoid overwhelming the LLM
        if len(diff_detail) > 6000:
            diff_detail = diff_detail[:6000] + "\n... (truncated)"

        prompt = f"""Review these recent code changes and extract learnings for the team.

COMMITS:
{log}

CHANGE SUMMARY:
{diff}

DIFF (key files):
{diff_detail}

Based on these changes, identify:
1. DECISIONS: What architectural or design decisions were made?
2. PATTERNS: What coding patterns or conventions were followed?
3. LESSONS: What could be improved or what risks should we watch?

Be specific and reference actual file names. Keep each point to 1-2 sentences."""

        await self._progress("learning_loop", 50, "AIIA analyzing changes")

        try:
            result = await self._aiia_request(
                "POST",
                "/v1/aiia/ask",
                {
                    "question": prompt,
                    "context": "You are AIIA, reviewing code changes to learn and grow smarter. Extract concrete, actionable insights.",
                    "n_results": 3,
                },
                timeout=90.0,
            )
            analysis = result.get("answer", "Analysis failed")
        except Exception as e:
            analysis = f"Analysis failed: {str(e)[:200]}"

        await self._progress("learning_loop", 75, "Storing learnings")

        # Store as structured memories
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Learning Loop {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: Reviewed {len(commits)} commits. {analysis[:500]}",
                "category": "patterns",
                "source": "task:learning_loop",
            },
        )

        # Create review actions if AIIA detected risks or issues
        if self.action_queue and analysis:
            lower = analysis.lower()
            risk_keywords = [
                "risk",
                "concern",
                "issue",
                "problem",
                "bug",
                "vulnerability",
                "breaking",
                "deprecated",
            ]
            if any(kw in lower for kw in risk_keywords):
                self.action_queue.create_action(
                    action_type="review",
                    severity="info",
                    title=f"Code review: {len(commits)} recent commits",
                    description=analysis[:500],
                    source_task="learning_loop",
                )
                await self.broadcast(
                    "new_action",
                    {"count": 1, "source": "learning_loop"},
                )

        self._extra["last_learned_sha"] = current_sha

        await self._progress("learning_loop", 100, "Complete")
        summary = f"Analyzed {len(commits)} commits, extracted learnings"
        return (summary, analysis)

    async def _task_test_runner(self) -> str:
        """Run test suite and track pass/fail trends."""
        await self._progress("test_runner", 5, "Discovering test files")

        repo = Path(self.repo_path)

        # Find test files
        test_files = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    test_files.append(os.path.join(root, f))
                elif f.endswith("_test.py"):
                    test_files.append(os.path.join(root, f))

        if not test_files:
            await self._progress("test_runner", 100, "Complete")
            return "No test files found"

        await self._progress("test_runner", 20, f"Running {len(test_files)} test files")

        # Try pytest first, fall back to unittest
        passed = 0
        failed = 0
        errors = 0
        output_lines = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                "-m",
                "pytest",
                "--tb=short",
                "-q",
                *test_files,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode()
            output_lines = output.strip().split("\n")

            # Parse pytest summary line like "5 passed, 2 failed"
            for line in reversed(output_lines):
                if "passed" in line or "failed" in line or "error" in line:
                    import re

                    p = re.search(r"(\d+) passed", line)
                    f = re.search(r"(\d+) failed", line)
                    e = re.search(r"(\d+) error", line)
                    if p:
                        passed = int(p.group(1))
                    if f:
                        failed = int(f.group(1))
                    if e:
                        errors = int(e.group(1))
                    break
        except asyncio.TimeoutError:
            output_lines = ["Tests timed out after 120s"]
            errors = 1
        except Exception as e:
            output_lines = [f"Test runner error: {str(e)[:200]}"]
            errors = 1

        await self._progress("test_runner", 85, "Recording results")

        report = "\n".join(output_lines[-30:])  # Last 30 lines

        # Track trends
        trends = self._extra.setdefault("test_trends", [])
        trends.append(
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "test_files": len(test_files),
            }
        )
        self._extra["test_trends"] = trends[-20:]

        # Emit test insights
        if failed > 0 or errors > 0:
            await self._emit_insight(
                "quality",
                "warn",
                f"Tests: {failed} failures, {errors} errors across {len(test_files)} files",
                source_task="test_runner",
            )
            # Create action items for test failures
            if self.action_queue:
                # Extract failing test names from output
                fail_lines = [l for l in output_lines if "FAILED" in l or "ERROR" in l]
                if fail_lines:
                    for line in fail_lines[:5]:
                        self.action_queue.create_action(
                            action_type="test_fix",
                            severity="warn",
                            title=f"Test failure: {line[:80]}",
                            description="\n".join(output_lines[-15:]),
                            source_task="test_runner",
                        )
                else:
                    self.action_queue.create_action(
                        action_type="test_fix",
                        severity="warn",
                        title=f"{failed} test failures, {errors} errors",
                        description="\n".join(output_lines[-20:]),
                        source_task="test_runner",
                    )
                await self.broadcast(
                    "new_action",
                    {"count": failed + errors, "source": "test_runner"},
                )
        elif passed > 0:
            await self._emit_insight(
                "quality",
                "success",
                f"All {passed} tests passing across {len(test_files)} files",
                source_task="test_runner",
            )

        # Store in AIIA
        status_word = (
            "all passing"
            if failed == 0 and errors == 0
            else f"{failed} failures, {errors} errors"
        )
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Test run {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: {passed} passed, {failed} failed, {errors} errors across {len(test_files)} test files — {status_word}",
                "category": "project",
                "source": "task:test_runner",
            },
        )

        await self._progress("test_runner", 100, "Complete")
        summary = f"{passed} passed, {failed} failed, {errors} errors"
        return (summary, report)

    async def _task_weekly_client_status(self) -> str:
        """Generate a client delivery report for the last 7 days."""
        await self._progress(
            "weekly_client_status", 5, "Getting client commits (7 days)"
        )

        # 1. Client-specific commits
        try:
            client_log = await self._run_git(
                "log",
                "--oneline",
                "--since=7 days ago",
                "--",
                "products/my-app/",
            )
            client_commits = [l for l in client_log.strip().split("\n") if l]
        except Exception:
            client_commits = []

        await self._progress("weekly_client_status", 15, "Getting platform commits")

        # 2. Platform commits that affect client infra
        try:
            platform_log = await self._run_git(
                "log",
                "--oneline",
                "--since=7 days ago",
                "--",
                "platform/",
            )
            platform_commits = [l for l in platform_log.strip().split("\n") if l]
        except Exception:
            platform_commits = []

        await self._progress("weekly_client_status", 25, "Parsing commit stats")

        # 3. Detailed stats for client changes
        try:
            stat_output = await self._run_git(
                "diff",
                "--stat",
                "--since=7 days ago",
                "HEAD@{7 days ago}..HEAD",
                "--",
                "products/my-app/",
            )
        except Exception:
            # Fallback: use shortlog
            stat_output = ""

        # Parse commit types
        features, fixes, chores, other = [], [], [], []
        for c in client_commits:
            subject = c.split(" ", 1)[1] if " " in c else c
            lower = subject.lower()
            if lower.startswith("feat"):
                features.append(subject)
            elif lower.startswith("fix"):
                fixes.append(subject)
            elif lower.startswith("chore") or lower.startswith("refactor"):
                chores.append(subject)
            else:
                other.append(subject)

        # File change count
        try:
            numstat = await self._run_git(
                "log",
                "--numstat",
                "--since=7 days ago",
                "--format=",
                "--",
                "products/my-app/",
            )
            additions, deletions, files_changed = 0, 0, set()
            for line in numstat.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    try:
                        additions += int(parts[0]) if parts[0] != "-" else 0
                        deletions += int(parts[1]) if parts[1] != "-" else 0
                        files_changed.add(parts[2])
                    except ValueError:
                        pass
        except Exception:
            additions, deletions, files_changed = 0, 0, set()

        await self._progress("weekly_client_status", 35, "Getting AIIA KB stats")

        # 4. AIIA knowledge stats
        try:
            aiia_status = await self._aiia_request(
                "GET", "/v1/aiia/status", timeout=10.0
            )
            kb_docs = aiia_status.get("knowledge", {}).get("knowledge_docs", 0)
            total_memories = aiia_status.get("memory", {}).get("total_memories", 0)
        except Exception:
            kb_docs, total_memories = 0, 0

        await self._progress("weekly_client_status", 45, "Building report data")

        # 5. Build structured data
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report_data = {
            "report_date": today,
            "period": "7 days",
            "product": "my-app",
            "client": "primary",
            "client_commits": len(client_commits),
            "platform_commits": len(platform_commits),
            "features": features,
            "fixes": fixes,
            "chores": chores,
            "other": other,
            "files_changed": len(files_changed),
            "additions": additions,
            "deletions": deletions,
            "kb_docs": kb_docs,
            "total_memories": total_memories,
        }

        await self._progress(
            "weekly_client_status", 60, "AIIA generating client summary"
        )

        # 6. Ask AIIA to generate client-facing summary
        # Keep prompt compact — strip commit prefixes, limit to 8 per category
        def _clean(subjects, limit=8):
            cleaned = []
            for s in subjects[:limit]:
                # Strip conventional commit prefix (feat(...): / fix: etc)
                s = s.split(":", 1)[-1].strip() if ":" in s else s
                # Remove special chars that might cause encoding issues
                s = s.replace("%", " percent").replace("\u2014", "-")
                cleaned.append(s[:80])
            return cleaned

        feat_list = _clean(features)
        fix_list = _clean(fixes)

        prompt = (
            f"Weekly status report for primary client, 7 days ending {today}. "
            f"Features shipped ({len(features)}): {'; '.join(feat_list)}. "
            f"Bug fixes ({len(fixes)}): {'; '.join(fix_list)}. "
            f"Also {len(chores)} maintenance and {len(platform_commits)} platform commits. "
            f"Metrics: {len(files_changed)} files changed, {additions} lines added, {deletions} removed. "
            f"Write a professional report: Executive Summary (2 sentences), "
            f"Features Shipped (bullets), Bug Fixes (bullets), Metrics (one line)."
        )

        try:
            result = await self._aiia_request(
                "POST",
                "/v1/aiia/ask",
                {
                    "question": prompt,
                    "context": "Generate a weekly client delivery report. Be concise and professional.",
                    "n_results": 1,
                },
                timeout=180.0,
            )
            report_text = result.get("answer", "Report generation failed.")
        except Exception as e:
            report_text = f"Report generation failed: {str(e)[:200]}"

        await self._progress("weekly_client_status", 80, "Saving report")

        # 7. Save report JSON
        reports_dir = Path(
            os.path.expanduser("~/.aiia/eq_data/weekly_reports")
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_data["generated_at"] = datetime.now(timezone.utc).isoformat()
        report_data["report_text"] = report_text

        report_path = reports_dir / f"{today}.json"
        report_path.write_text(json.dumps(report_data, indent=2))

        await self._progress("weekly_client_status", 90, "Storing in AIIA memory")

        # 8. Store summary in AIIA memory
        brief = report_text[:500] if len(report_text) > 500 else report_text
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Client Weekly Status {today}: {len(client_commits)} commits ({len(features)} features, {len(fixes)} fixes), +{additions}/-{deletions} lines, {len(files_changed)} files. {brief}",
                "category": "project",
                "source": "task:weekly_client_status",
            },
        )

        await self._emit_insight(
            "shipped",
            "info",
            f"Client weekly: {len(features)} features, {len(fixes)} fixes, +{additions}/-{deletions} lines",
            source_task="weekly_client_status",
        )

        # Track in _extra for delta comparison
        self._extra["weekly_client_last"] = {
            "date": today,
            "commits": len(client_commits),
            "features": len(features),
            "fixes": len(fixes),
        }

        await self._progress("weekly_client_status", 100, "Complete")
        summary = f"{len(client_commits)} commits ({len(features)} feat, {len(fixes)} fix), +{additions}/-{deletions} lines"
        return (summary, report_text)

    async def _task_cross_tenant_analytics(self) -> str:
        """Aggregate conversation patterns across tenants from PostgreSQL (read-only)."""
        db_url = os.getenv("CROSS_TENANT_DB_URL")
        if not db_url:
            await self._progress("cross_tenant_analytics", 100, "Complete")
            return "Skipped — CROSS_TENANT_DB_URL not configured"

        await self._progress("cross_tenant_analytics", 10, "Connecting to database")

        try:
            import asyncpg
        except ImportError:
            await self._progress("cross_tenant_analytics", 100, "Complete")
            return "Skipped — asyncpg not installed"

        try:
            conn = await asyncpg.connect(db_url)
        except Exception as e:
            await self._progress("cross_tenant_analytics", 100, "Complete")
            return f"DB connection failed: {str(e)[:100]}"

        try:
            await self._progress(
                "cross_tenant_analytics", 20, "Querying conversation stats"
            )

            # Total conversations per tenant (last 7 days)
            rows = await conn.fetch("""
                SELECT tenant_id, COUNT(*) as conv_count,
                       COUNT(DISTINCT user_id) as user_count
                FROM conversations
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY tenant_id
                ORDER BY conv_count DESC
            """)

            await self._progress(
                "cross_tenant_analytics", 40, "Querying message volume"
            )

            # Message volume per tenant
            msg_rows = await conn.fetch("""
                SELECT c.tenant_id, COUNT(m.id) as msg_count
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.created_at > NOW() - INTERVAL '7 days'
                GROUP BY c.tenant_id
                ORDER BY msg_count DESC
            """)

            await self._progress("cross_tenant_analytics", 60, "Building analytics")

            # Build analytics summary
            tenant_stats = {}
            for row in rows:
                tid = row["tenant_id"]
                tenant_stats[tid] = {
                    "conversations": row["conv_count"],
                    "users": row["user_count"],
                    "messages": 0,
                }
            for row in msg_rows:
                tid = row["tenant_id"]
                if tid in tenant_stats:
                    tenant_stats[tid]["messages"] = row["msg_count"]

            await self._progress(
                "cross_tenant_analytics", 70, "Asking AIIA for insights"
            )

            # Build summary text
            stats_text = "\n".join(
                f"  {tid}: {s['conversations']} conversations, {s['users']} users, {s['messages']} messages"
                for tid, s in tenant_stats.items()
            )

            # Ask AIIA for insights
            prompt = (
                f"Cross-tenant analytics for the past 7 days:\n{stats_text}\n\n"
                "Identify patterns: Which tenants are most active? Any anomalies? "
                "Suggest optimizations. Keep it to 3-4 sentences."
            )

            try:
                result = await self._aiia_request(
                    "POST",
                    "/v1/aiia/ask",
                    {
                        "question": prompt,
                        "context": "You are analyzing cross-tenant usage patterns for the AIIA platform.",
                        "n_results": 2,
                    },
                    timeout=60.0,
                )
                insights = result.get("answer", "No insights generated")
            except Exception:
                insights = "AIIA analysis unavailable"

            await self._progress("cross_tenant_analytics", 85, "Storing in memory")

            # Store in AIIA memory
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            total_convs = sum(s["conversations"] for s in tenant_stats.values())
            total_msgs = sum(s["messages"] for s in tenant_stats.values())

            await self._aiia_request(
                "POST",
                "/v1/aiia/remember",
                {
                    "fact": (
                        f"Cross-tenant analytics {today}: {len(tenant_stats)} tenants, "
                        f"{total_convs} conversations, {total_msgs} messages (7d). {insights[:300]}"
                    ),
                    "category": "project",
                    "source": "task:cross_tenant_analytics",
                },
            )

            await self._progress("cross_tenant_analytics", 100, "Complete")
            summary = f"{len(tenant_stats)} tenants, {total_convs} conversations, {total_msgs} messages (7d)"
            return (summary, f"Stats:\n{stats_text}\n\nInsights:\n{insights}")

        finally:
            await conn.close()

    async def _task_security_scan(self) -> str:
        """Check Dependabot alerts, CI status, dependency vulnerabilities via gh CLI."""
        await self._progress("security_scan", 5, "Checking Dependabot alerts")

        # 1. Dependabot alerts
        alerts_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        total_open = 0
        alert_details = []

        try:
            alerts_json = await self._run_command(
                "gh",
                "api",
                "repos/OWNER/AIIA/dependabot/alerts",
                "--jq",
                '[.[] | select(.state=="open") | {severity: .security_advisory.severity, package: .dependency.package.name, summary: .security_advisory.summary}]',
                timeout=20.0,
            )
            if alerts_json.strip():
                alerts = json.loads(alerts_json)
                total_open = len(alerts)
                for alert in alerts:
                    sev = (alert.get("severity") or "unknown").lower()
                    if sev in alerts_by_severity:
                        alerts_by_severity[sev] += 1
                    pkg = alert.get("package", "?")
                    summary = alert.get("summary", "")[:80]
                    alert_details.append(f"{sev}: {pkg} — {summary}")
        except Exception as e:
            logger.warning(f"Dependabot check failed: {e}")

        await self._progress("security_scan", 30, "Checking CI status")

        # 2. Recent CI runs
        ci_runs = []
        try:
            runs_json = await self._run_command(
                "gh",
                "api",
                "repos/OWNER/AIIA/actions/runs?per_page=5",
                "--jq",
                "[.workflow_runs[:5] | .[] | {status: .status, conclusion: .conclusion, name: .name, created_at: .created_at}]",
                timeout=20.0,
            )
            if runs_json.strip():
                ci_runs = json.loads(runs_json)
        except Exception as e:
            logger.warning(f"CI status check failed: {e}")

        await self._progress("security_scan", 55, "Building security snapshot")

        # 3. Build security snapshot
        ci_passing = all(
            r.get("conclusion") == "success"
            for r in ci_runs
            if r.get("status") == "completed"
        )
        ci_status = "passing" if ci_passing and ci_runs else "unknown"

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependabot": {
                "total_open": total_open,
                "by_severity": alerts_by_severity,
                "details": alert_details[:10],
            },
            "ci": {
                "status": ci_status,
                "recent_runs": ci_runs,
            },
        }

        self._extra["security_snapshot"] = snapshot

        # Track trends
        trends = self._extra.setdefault("security_trends", [])
        trends.append(
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "open_alerts": total_open,
                "critical": alerts_by_severity["critical"],
                "high": alerts_by_severity["high"],
                "ci_passing": ci_passing,
            }
        )
        self._extra["security_trends"] = trends[-20:]

        await self._progress("security_scan", 75, "Emitting insights")

        # 4. Emit insights
        if total_open > 0:
            severity_parts = []
            for sev in ["critical", "high", "medium", "low"]:
                count = alerts_by_severity[sev]
                if count > 0:
                    severity_parts.append(f"{count} {sev}")
            await self._emit_insight(
                "security",
                "warn" if alerts_by_severity["critical"] == 0 else "error",
                f"{total_open} Dependabot alerts open ({', '.join(severity_parts)})",
                detail="\n".join(alert_details[:5]),
                source_task="security_scan",
            )
            # Create action items for critical/high CVEs
            if self.action_queue:
                for detail in alert_details:
                    detail_sev = detail.split(":")[0].strip().lower()
                    if detail_sev in ("critical", "high"):
                        self.action_queue.create_action(
                            action_type="security_fix",
                            severity="error" if detail_sev == "critical" else "warn",
                            title=f"Dependabot: {detail[:80]}",
                            description=detail,
                            source_task="security_scan",
                        )
                await self.broadcast(
                    "new_action",
                    {
                        "count": alerts_by_severity["critical"]
                        + alerts_by_severity["high"],
                        "source": "security_scan",
                    },
                )
        else:
            await self._emit_insight(
                "security",
                "success",
                "No open Dependabot alerts",
                source_task="security_scan",
            )

        if ci_runs:
            await self._emit_insight(
                "security",
                "success" if ci_passing else "warn",
                f"CI {ci_status}" + (" (last 5 runs)" if ci_passing else ""),
                source_task="security_scan",
            )

        await self._progress("security_scan", 90, "Storing in AIIA memory")

        # 5. Store in AIIA
        await self._aiia_request(
            "POST",
            "/v1/aiia/remember",
            {
                "fact": f"Security scan {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: {total_open} Dependabot alerts (critical={alerts_by_severity['critical']}, high={alerts_by_severity['high']}), CI {ci_status}",
                "category": "project",
                "source": "task:security_scan",
            },
        )

        await self._progress("security_scan", 100, "Complete")

        report_lines = [
            f"Security Scan {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}",
            f"Dependabot: {total_open} open alerts",
            f"  Critical: {alerts_by_severity['critical']}",
            f"  High: {alerts_by_severity['high']}",
            f"  Medium: {alerts_by_severity['medium']}",
            f"  Low: {alerts_by_severity['low']}",
            f"CI Status: {ci_status}",
        ]
        if alert_details:
            report_lines.append("")
            report_lines.append("Open Alerts:")
            report_lines.extend(f"  {d}" for d in alert_details[:10])
        if ci_runs:
            report_lines.append("")
            report_lines.append("Recent CI Runs:")
            for run in ci_runs:
                report_lines.append(
                    f"  {run.get('name', '?')}: {run.get('conclusion', run.get('status', '?'))}"
                )

        report = "\n".join(report_lines)
        summary = f"{total_open} alerts ({alerts_by_severity['critical']}C/{alerts_by_severity['high']}H), CI {ci_status}"
        return (summary, report)

    async def _task_ci_monitor(self) -> str:
        """Poll GitHub Actions for recent CI failures, create action items."""
        await self._progress("ci_monitor", 10, "Polling GitHub Actions")

        ci_runs = []
        try:
            runs_json = await self._run_command(
                "gh",
                "api",
                "repos/OWNER/AIIA/actions/runs?per_page=3",
                "--jq",
                "[.workflow_runs[:3] | .[] | {id: .id, status: .status, conclusion: .conclusion, name: .name, created_at: .created_at, html_url: .html_url, head_branch: .head_branch}]",
                timeout=20.0,
            )
            if runs_json.strip():
                ci_runs = json.loads(runs_json)
        except Exception as e:
            logger.warning(f"CI monitor check failed: {e}")
            await self._progress("ci_monitor", 100, "Complete")
            return f"CI check failed: {str(e)[:100]}"

        await self._progress("ci_monitor", 50, f"Checking {len(ci_runs)} runs")

        if not ci_runs:
            await self._progress("ci_monitor", 100, "Complete")
            return "No CI runs found"

        failed_runs = [
            r
            for r in ci_runs
            if r.get("status") == "completed" and r.get("conclusion") == "failure"
        ]

        await self._progress("ci_monitor", 70, "Processing results")

        if failed_runs and self.action_queue:
            # Check how long failures have been happening
            ci_fail_history = self._extra.get("ci_fail_history", {})
            now = datetime.now(timezone.utc)

            for run in failed_runs:
                run_name = run.get("name", "unknown")
                branch = run.get("head_branch", "?")
                run_url = run.get("html_url", "")

                # Track when we first saw this workflow failing
                fail_key = f"{run_name}:{branch}"
                if fail_key not in ci_fail_history:
                    ci_fail_history[fail_key] = now.isoformat()

                # Check if failing > 24h -> escalate to critical
                first_fail = datetime.fromisoformat(ci_fail_history[fail_key])
                if first_fail.tzinfo is None:
                    first_fail = first_fail.replace(tzinfo=timezone.utc)
                hours_failing = (now - first_fail).total_seconds() / 3600
                severity = "error" if hours_failing > 24 else "warn"

                self.action_queue.create_action(
                    action_type="ci_fix",
                    severity=severity,
                    title=f"CI failing: {run_name} on {branch}",
                    description=f"GitHub Actions workflow '{run_name}' failed on branch '{branch}'. "
                    + (
                        f"Failing for {hours_failing:.0f}h. "
                        if hours_failing > 1
                        else ""
                    )
                    + (f"URL: {run_url}" if run_url else ""),
                    source_task="ci_monitor",
                )

            self._extra["ci_fail_history"] = ci_fail_history
            await self.broadcast(
                "new_action",
                {"count": len(failed_runs), "source": "ci_monitor"},
            )
        elif not failed_runs:
            # Clear fail history for passing workflows
            ci_fail_history = self._extra.get("ci_fail_history", {})
            passing_names = {
                f"{r.get('name', '')}:{r.get('head_branch', '')}"
                for r in ci_runs
                if r.get("conclusion") == "success"
            }
            for key in list(ci_fail_history.keys()):
                if key in passing_names:
                    del ci_fail_history[key]
            self._extra["ci_fail_history"] = ci_fail_history

        # Emit insight
        if failed_runs:
            names = [r.get("name", "?") for r in failed_runs]
            await self._emit_insight(
                "quality",
                "warn",
                f"CI: {len(failed_runs)} failing ({', '.join(names)})",
                source_task="ci_monitor",
            )
        else:
            await self._emit_insight(
                "quality",
                "success",
                f"CI: all {len(ci_runs)} recent runs passing",
                source_task="ci_monitor",
            )

        await self._progress("ci_monitor", 100, "Complete")
        passing = len(ci_runs) - len(failed_runs)
        summary = f"{passing} passing, {len(failed_runs)} failing"
        return summary
