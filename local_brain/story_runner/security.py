"""
Security module for AIIA Story Runner.

Bash command allowlist and filesystem restrictions.
Defense-in-depth: OS sandbox + command allowlist + path restrictions.
"""

import re
from typing import Set

# Commands the coding agent is allowed to run
ALLOWED_BASH_COMMANDS: Set[str] = {
    # File inspection
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "file",
    "du",
    # Python
    "python",
    "python3",
    "pip",
    "pip3",
    "uv",
    "uvx",
    "pytest",
    "ruff",
    "mypy",
    "black",
    "isort",
    "bandit",
    # Node.js
    "npm",
    "npx",
    "node",
    "tsc",
    # Version control
    "git",
    # Process management
    "ps",
    "lsof",
    "sleep",
    "pkill",
    "kill",
    "timeout",
    # Utilities
    "echo",
    "printf",
    "date",
    "env",
    "which",
    "dirname",
    "basename",
    "mkdir",
    "cp",
    "mv",
    "rm",
    "touch",
    "chmod",
    "sort",
    "uniq",
    "sed",
    "awk",
    "tr",
    "cut",
    "xargs",
    "diff",
    # Build tools
    "make",
    "cargo",
    "go",
}

# Commands that are NEVER allowed
BLOCKED_COMMANDS: Set[str] = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",  # no network
    "docker",
    "podman",  # no containers
    "sudo",
    "su",
    "doas",  # no privilege escalation
    "rm -rf /",
    "mkfs",
    "dd",  # no destruction
    "shutdown",
    "reboot",
    "halt",  # no system control
}

# Patterns that should be blocked
BLOCKED_PATTERNS = [
    r"git\s+push.*--force",  # no force push
    r"git\s+push.*main\b",  # no push to main (must go through PR)
    r"git\s+push.*master\b",
    r"rm\s+-rf\s+/",  # no recursive delete of root
    r">\s*/etc/",  # no writing to system dirs
    r"curl.*\|.*sh",  # no pipe-to-shell
]


def validate_bash_command(command: str) -> tuple[bool, str]:
    """
    Validate a bash command against the allowlist.

    Returns:
        (allowed: bool, reason: str)
    """
    if not command or not command.strip():
        return True, ""

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Blocked pattern: {pattern}"

    # Extract first command (handle pipes, chains)
    # Split on pipe, &&, ||, ; and check each segment
    segments = re.split(r"[|&;]", command)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Get the base command (handle env vars, cd, etc.)
        words = segment.split()
        base_cmd = None
        for word in words:
            # Skip env var assignments (FOO=bar)
            if "=" in word and not word.startswith("-"):
                continue
            # Skip cd (always allowed)
            if word == "cd":
                base_cmd = "cd"
                break
            # Skip common prefixes
            if word in (
                "set",
                "export",
                "local",
                "readonly",
                "SKIP=frontend-typecheck",
            ):
                continue
            base_cmd = word.split("/")[-1]  # handle full paths
            break

        if base_cmd and base_cmd not in ALLOWED_BASH_COMMANDS and base_cmd != "cd":
            # Check if it's a subcommand of an allowed tool
            if not any(
                base_cmd.startswith(allowed) for allowed in ALLOWED_BASH_COMMANDS
            ):
                return False, f"Command not in allowlist: {base_cmd}"

    return True, ""
