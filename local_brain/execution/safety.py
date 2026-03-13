from enum import Enum
from pathlib import PurePosixPath
from typing import Any


class SafetyTier(Enum):
    AUTO = "auto"
    SUPERVISED = "supervised"
    GATED = "gated"


# (action_type, severity) -> tier
# None severity = wildcard match
_TIER_MAP: dict[tuple[str, str | None], SafetyTier] = {
    ("lint_fix", None): SafetyTier.AUTO,
    ("verify_lint", None): SafetyTier.AUTO,
    ("verify_test", None): SafetyTier.AUTO,
    ("verify_security", None): SafetyTier.AUTO,
    ("commit", None): SafetyTier.SUPERVISED,
    ("test_fix", "warn"): SafetyTier.SUPERVISED,
    ("test_fix", "error"): SafetyTier.SUPERVISED,
    ("ci_fix", "warn"): SafetyTier.SUPERVISED,
    ("ci_fix", "error"): SafetyTier.SUPERVISED,
    ("security_fix", "error"): SafetyTier.GATED,
    ("security_fix", "critical"): SafetyTier.GATED,
    ("tech_debt", None): SafetyTier.SUPERVISED,
    ("review", None): SafetyTier.GATED,
}

_FORBIDDEN_PATHS = [
    "products/my-app/backend/main.py",
    ".env",
]

_FORBIDDEN_EXTENSIONS = {".pem", ".key"}

_FORBIDDEN_PATH_SEGMENTS = {"migration"}

_FORBIDDEN_EXACT = {"render.yaml"}

ALLOWED_GIT_OPS = [
    "add",
    "commit",
    "status",
    "diff",
    "log",
    "branch",
    "checkout",
    "switch",
]


def _is_forbidden(path: str) -> bool:
    p = PurePosixPath(path)

    if path in _FORBIDDEN_PATHS or p.name in _FORBIDDEN_EXACT:
        return True

    if p.suffix in _FORBIDDEN_EXTENSIONS:
        return True

    if p.name == ".env" or p.name.startswith(".env."):
        return True

    for part in p.parts:
        if part.lower() in _FORBIDDEN_PATH_SEGMENTS:
            return True

    return False


class SafetyGate:
    def __init__(
        self,
        max_concurrent: int = 1,
        max_files_per_action: int = 20,
    ):
        self._kill_switch = False
        self.max_concurrent = max_concurrent
        self.max_files_per_action = max_files_per_action

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    @kill_switch.setter
    def kill_switch(self, value: bool) -> None:
        self._kill_switch = value

    def get_tier(self, action: dict[str, Any]) -> SafetyTier:
        action_type = action.get("type", action.get("action_type", ""))
        severity = action.get("severity")

        # Try exact match first
        key = (action_type, severity)
        if key in _TIER_MAP:
            return _TIER_MAP[key]

        # Try wildcard (None severity = any)
        wildcard_key = (action_type, None)
        if wildcard_key in _TIER_MAP:
            return _TIER_MAP[wildcard_key]

        return SafetyTier.GATED

    def can_execute(self, action: dict[str, Any]) -> tuple[bool, str]:
        if self._kill_switch:
            return False, "kill switch engaged"

        files = action.get("files_affected", action.get("files", []))
        if len(files) > self.max_files_per_action:
            return (
                False,
                f"action touches {len(files)} files, max is"
                f" {self.max_files_per_action}",
            )

        ok, violations = self.validate_files(files)
        if not ok:
            return False, f"forbidden paths: {', '.join(violations)}"

        return True, ""

    def check_git_op(self, op: str) -> bool:
        return op.lower() in ALLOWED_GIT_OPS

    def validate_files(self, files: list[str]) -> tuple[bool, list[str]]:
        violations = [f for f in files if _is_forbidden(f)]
        return (len(violations) == 0, violations)
