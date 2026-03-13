from __future__ import annotations

import time
from dataclasses import dataclass

from .subprocess_pool import SubprocessPool


@dataclass
class VerificationResult:
    verified: bool | None  # None = can't verify
    reason: str
    output: str = ""
    duration_ms: int = 0


class Verifier:
    def __init__(self, pool: SubprocessPool, repo_path: str) -> None:
        self._pool = pool
        self._repo_path = repo_path

    async def verify(self, action: dict) -> VerificationResult:
        start = time.monotonic()
        try:
            action_type = action.get("type", "")
            handler = {
                "lint_fix": self._verify_lint,
                "verify_lint": self._verify_lint,
                "test_fix": self._verify_test,
                "verify_test": self._verify_test,
                "security_fix": self._verify_security,
                "verify_security": self._verify_security,
                "ci_fix": self._verify_ci,
                "tech_debt": self._verify_tech_debt,
                "commit": self._verify_commit,
            }.get(action_type)

            if handler is None:
                return VerificationResult(
                    verified=None,
                    reason=f"No verifier for type '{action_type}'",
                    duration_ms=_elapsed_ms(start),
                )

            result = await handler(action)
            result.duration_ms = _elapsed_ms(start)
            return result
        except Exception as e:
            return VerificationResult(
                verified=False,
                reason=str(e),
                duration_ms=_elapsed_ms(start),
            )

    async def _verify_lint(self, action: dict) -> VerificationResult:
        files = action.get("files_affected", [])
        if not files:
            files = ["local_brain/execution/"]

        cmd = ["ruff", "check"] + files
        result = await self._pool.run(cmd, cwd=self._repo_path, timeout=30)
        return VerificationResult(
            verified=result.returncode == 0,
            reason=(
                "Lint check passed"
                if result.returncode == 0
                else "Lint check still failing"
            ),
            output=_truncate_tail(result.stdout + result.stderr),
        )

    async def _verify_test(self, action: dict) -> VerificationResult:
        files = action.get("files_affected", [])
        test_files = []
        for f in files:
            if _is_test_file(f):
                test_files.append(f)
            else:
                # Try to find corresponding test file
                found = _resolve_test_target(f)
                if found:
                    test_files.append(found)

        cmd = ["python", "-m", "pytest"]
        if test_files:
            cmd += test_files
        cmd += ["-x", "-v", "--timeout=120"]

        result = await self._pool.run(cmd, cwd=self._repo_path, timeout=180)
        return VerificationResult(
            verified=result.returncode == 0,
            reason=(
                "Tests passed" if result.returncode == 0 else "Tests still failing"
            ),
            output=_truncate_tail(result.stdout + result.stderr),
        )

    async def _verify_security(self, action: dict) -> VerificationResult:
        scanner = action.get("scanner")
        if scanner:
            cmd = [scanner] + action.get("scanner_args", [])
        else:
            cmd = ["brain", "scan", "-q"]

        result = await self._pool.run(cmd, cwd=self._repo_path, timeout=180)
        return VerificationResult(
            verified=result.returncode == 0,
            reason=(
                "Security scan passed"
                if result.returncode == 0
                else "Security scan still reports issues"
            ),
            output=_truncate_tail(result.stdout + result.stderr),
        )

    async def _verify_ci(self, action: dict) -> VerificationResult:
        return VerificationResult(
            verified=None,
            reason="CI verification requires push",
        )

    async def _verify_commit(self, action: dict) -> VerificationResult:
        result = await self._pool.run(
            ["git", "status", "--porcelain"],
            cwd=self._repo_path,
            timeout=10,
        )
        clean = result.returncode == 0 and not result.stdout.strip()
        return VerificationResult(
            verified=clean,
            reason="Working tree clean" if clean else "Uncommitted changes remain",
            output=_truncate_tail(result.stdout),
        )

    async def _verify_tech_debt(self, action: dict) -> VerificationResult:
        files = action.get("files_affected", [])
        if not files:
            return VerificationResult(
                verified=None,
                reason="No files_affected for tech debt verification",
            )

        lint_cmd = ["ruff", "check"] + files
        lint_result = await self._pool.run(lint_cmd, cwd=self._repo_path, timeout=30)
        if lint_result.returncode != 0:
            return VerificationResult(
                verified=False,
                reason="Lint still failing after tech debt fix",
                output=_truncate_tail(lint_result.stdout + lint_result.stderr),
            )

        test_files = [f for f in files if _is_test_file(f)]
        if not test_files:
            return VerificationResult(
                verified=True,
                reason="Lint passed, no test files to verify",
                output=_truncate_tail(lint_result.stdout + lint_result.stderr),
            )

        test_cmd = ["python", "-m", "pytest"] + test_files + ["-x", "-v"]
        test_result = await self._pool.run(test_cmd, cwd=self._repo_path, timeout=120)
        combined = lint_result.stdout + test_result.stdout + test_result.stderr
        return VerificationResult(
            verified=test_result.returncode == 0,
            reason=(
                "Lint + tests passed"
                if test_result.returncode == 0
                else "Tests still failing after tech debt fix"
            ),
            output=_truncate_tail(combined),
        )


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    return name.startswith("test_") or name.endswith("_test.py")


def _resolve_test_target(source_path: str) -> str | None:
    """Try to find a test file for a source file by convention."""
    import os

    if not source_path.endswith(".py"):
        return None
    name = source_path.rsplit("/", 1)[-1] if "/" in source_path else source_path
    directory = source_path.rsplit("/", 1)[0] if "/" in source_path else "."

    candidates = [
        f"{directory}/test_{name}",
        f"{directory}/tests/test_{name}",
        f"tests/test_{name}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _truncate_tail(text: str, max_len: int = 1000) -> str:
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
