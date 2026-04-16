"""
SubprocessExecutor — wraps a shell command as an A2A agent.

Used for read-only dev tooling where the agent's "work" is running a CLI
and returning its stdout. Uses asyncio.create_subprocess_exec so the event
loop isn't blocked while the tool runs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Sequence

from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.schema import Message, TextPart

logger = logging.getLogger("aplora.a2a.subprocess_executor")


class SubprocessExecutor(AgentExecutor):
    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        artifact_name: str = "stdout",
        include_stderr: bool = True,
        cwd: Optional[str] = None,
    ) -> None:
        if not command:
            raise ValueError("command must be a non-empty sequence")
        self._command = list(command)
        self._timeout = timeout_seconds
        self._artifact_name = artifact_name
        self._include_stderr = include_stderr
        self._cwd = cwd

    async def execute(self, message: Message) -> ExecutorResult:
        logger.info("running subprocess agent: %s", " ".join(self._command))
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"subprocess timed out after {self._timeout}s: {' '.join(self._command)}"
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"command not found: {exc}") from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            detail = stderr or stdout or f"exit {proc.returncode}"
            raise RuntimeError(f"subprocess failed: {detail}")

        body = stdout
        if self._include_stderr and stderr:
            body = f"{stdout}\n\n[stderr]\n{stderr}"

        return ExecutorResult(
            parts=[TextPart(text=body or "(no output)")],
            artifact_name=self._artifact_name,
            artifact_description=f"Output of `{' '.join(self._command)}`",
            metadata={"returncode": proc.returncode},
        )
