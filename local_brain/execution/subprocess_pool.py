from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger("aiia.execution.pool")


class ExecutionTimeout(Exception):
    pass


@dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


class SubprocessPool:
    def __init__(self, max_concurrent: int = 1, default_timeout: float = 120):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: dict[str, asyncio.subprocess.Process] = {}
        self._default_timeout = default_timeout

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def run(
        self,
        cmd: list[str],
        cwd: str,
        timeout: float | None = None,
        env: dict | None = None,
    ) -> SubprocessResult:
        async with self._semaphore:
            start = time.monotonic()
            proc_id = uuid.uuid4().hex[:8]
            # Ensure user tool paths are available (launchd strips PATH)
            home = os.path.expanduser("~")
            extra_paths = [
                f"{home}/.local/bin",
                f"{home}/.aiia/venv/bin",
                "/usr/local/bin",
                "/opt/homebrew/bin",
            ]
            base_path = os.environ.get("PATH", "/usr/bin:/bin")
            full_path = ":".join(extra_paths) + ":" + base_path
            merged_env = {
                **os.environ,
                "PATH": full_path,
                **(env or {}),
            }

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
            self._active[proc_id] = proc

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout or self._default_timeout,
                )
                elapsed = int((time.monotonic() - start) * 1000)
                return SubprocessResult(
                    returncode=proc.returncode or 0,
                    stdout=stdout.decode(errors="replace"),
                    stderr=stderr.decode(errors="replace"),
                    duration_ms=elapsed,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = int((time.monotonic() - start) * 1000)
                logger.warning(
                    f"Process {proc_id} timed out after"
                    f" {timeout or self._default_timeout}s: {cmd[0]}"
                )
                raise ExecutionTimeout(
                    f"Command timed out after"
                    f" {timeout or self._default_timeout}s: {' '.join(cmd[:3])}"
                )
            finally:
                self._active.pop(proc_id, None)

    async def run_claude_code(
        self,
        prompt: str,
        cwd: str,
        timeout: float = 600,
    ) -> SubprocessResult:
        cmd = [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            prompt,
        ]
        return await self.run(cmd, cwd=cwd, timeout=timeout)

    async def kill_all(self) -> int:
        count = 0
        for proc_id, proc in list(self._active.items()):
            try:
                proc.kill()
                await proc.wait()
                count += 1
                logger.warning(f"Killed subprocess {proc_id}")
            except ProcessLookupError:
                pass
        self._active.clear()
        return count
