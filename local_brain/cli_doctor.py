"""
⌬ aiia doctor — environment + service health diagnostics.

Runs a battery of checks against the local environment, services, models,
and storage. Surfaces problems with names + actionable hints instead of
raw stack traces. Designed to be the FIRST thing a user runs when
"something's wrong."

Each check returns a Result with a level (`ok` / `warn` / `error`), a
short message, and an optional hint. The overall exit code is:

    0  every required check passes
    1  one or more required checks failed

Service-not-running is `warn`, not `error` — `aiia doctor` is a snapshot,
not an enforcement gate. The user can decide what to start.

Hint discipline: every non-ok status should include a concrete next step
(a command to run, a file to inspect, a URL to open). If a check fires
and the hint isn't actionable, the check is doing its user wrong.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------

Level = str  # "ok" | "warn" | "error"


@dataclass
class Result:
    name: str
    level: Level
    message: str
    hint: str | None = None

    @property
    def is_blocking(self) -> bool:
        return self.level == "error"


# ----------------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------------

MIN_PYTHON = (3, 10)


def check_python_version() -> Result:
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PYTHON:
        return Result(
            "python",
            "ok",
            f"Python {v.major}.{v.minor}.{v.micro}",
        )
    return Result(
        "python",
        "error",
        f"Python {v.major}.{v.minor}.{v.micro} (need ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        hint=(
            f"AIIA requires Python ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}. "
            "Install with pyenv, asdf, or your distro package manager, then "
            "re-create the venv."
        ),
    )


def check_local_brain_importable() -> Result:
    try:
        import local_brain  # noqa: F401 — import side-effect is the check

        return Result("local_brain package", "ok", "importable")
    except ImportError as e:
        return Result(
            "local_brain package",
            "error",
            f"not importable: {e}",
            hint="Run `pip install -e .` at the repo root.",
        )


def _http_get_status(url: str, timeout: float = 3.0) -> tuple[int, dict | None]:
    """GET a URL. Returns (status, parsed_json_or_none).

    Status -1 means the connection failed entirely (service down / network).
    Status 0 means the response wasn't JSON.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            try:
                return status, json.loads(resp.read())
            except json.JSONDecodeError:
                return 0, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError):
        return -1, None


def check_brain() -> Result:
    url = os.environ.get(
        "LOCAL_BRAIN_URL",
        f"http://localhost:{os.environ.get('LOCAL_BRAIN_PORT', '8100')}",
    )
    status, body = _http_get_status(f"{url}/health")
    if status == -1:
        return Result(
            "Brain API",
            "warn",
            f"unreachable at {url}",
            hint=("Start the Brain in another shell: `uvicorn local_brain.main:app --port 8100`"),
        )
    if status >= 400:
        return Result(
            "Brain API",
            "warn",
            f"responded HTTP {status} from {url}/health",
            hint="Check Brain logs; /health should return 200.",
        )
    version = (body or {}).get("version", "?")
    return Result("Brain API", "ok", f"up at {url} (v{version})")


def check_command_center() -> Result:
    port = os.environ.get("COMMAND_CENTER_PORT", "8200")
    url = f"http://localhost:{port}"
    status, _ = _http_get_status(f"{url}/health")
    if status == -1:
        return Result(
            "Command Center",
            "warn",
            f"unreachable at {url}",
            hint=(
                "Start the Command Center in another shell: "
                "`python -m local_brain.command_center.server`"
            ),
        )
    if status >= 400:
        return Result(
            "Command Center",
            "warn",
            f"responded HTTP {status} from {url}/health",
        )
    return Result("Command Center", "ok", f"up at {url}")


def check_ollama() -> Result:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    status, body = _http_get_status(f"{url}/api/tags")
    if status == -1:
        return Result(
            "Ollama",
            "error",
            f"unreachable at {url}",
            hint=(
                "Ollama is required for local LLM routing. Install from "
                "https://ollama.com/download and start it (`ollama serve`)."
            ),
        )
    if status >= 400:
        return Result(
            "Ollama",
            "warn",
            f"responded HTTP {status} from {url}/api/tags",
        )
    models = [m.get("name", "?") for m in (body or {}).get("models", [])]
    if not models:
        return Result(
            "Ollama",
            "warn",
            f"up at {url} but no models pulled",
            hint=("Pull at least the default model: `ollama pull llama3.1:8b-instruct-q8_0`"),
        )
    return Result(
        "Ollama",
        "ok",
        f"up at {url} ({len(models)} model{'s' if len(models) != 1 else ''})",
    )


def check_default_model() -> Result:
    """Check that AIIA_DEFAULT_MODEL is pulled in Ollama."""
    default_model = os.environ.get("AIIA_DEFAULT_MODEL", "llama3.1:8b-instruct-q8_0")
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    status, body = _http_get_status(f"{url}/api/tags")
    if status != 200 or not body:
        # Ollama check will surface this; don't double-report.
        return Result(
            "Default model",
            "warn",
            f"can't verify — Ollama unreachable at {url}",
        )
    pulled = {m.get("name", "") for m in body.get("models", [])}
    if default_model in pulled:
        return Result("Default model", "ok", f"{default_model} present in Ollama")
    return Result(
        "Default model",
        "warn",
        f"AIIA_DEFAULT_MODEL={default_model} not pulled",
        hint=f"`ollama pull {default_model}`",
    )


def check_vault_dir() -> Result:
    """The EQ Brain stores ChromaDB + memory at this path."""
    raw = os.environ.get("EQ_BRAIN_DATA_DIR", "~/.aiia/eq_data")
    path = Path(os.path.expanduser(raw))
    if not path.exists():
        return Result(
            "Vault dir",
            "warn",
            f"{path} does not exist (will be created on first run)",
        )
    if not path.is_dir():
        return Result(
            "Vault dir",
            "error",
            f"{path} exists but is not a directory",
            hint=f"Remove or rename {path} and re-run.",
        )
    # Check writability without actually writing.
    if not os.access(path, os.W_OK):
        return Result(
            "Vault dir",
            "error",
            f"{path} not writable",
            hint=f"Fix permissions: `chmod u+w {path}`",
        )
    return Result("Vault dir", "ok", f"{path} writable")


def check_chromadb_importable() -> Result:
    try:
        import chromadb  # noqa: F401

        return Result("chromadb", "ok", "importable")
    except ImportError:
        return Result(
            "chromadb",
            "error",
            "chromadb not installed",
            hint="`pip install -e .` should pull it; check for install errors.",
        )


def check_disk_space() -> Result:
    """Surface disk space remaining on the vault dir's filesystem."""
    raw = os.environ.get("EQ_BRAIN_DATA_DIR", "~/.aiia")
    path = Path(os.path.expanduser(raw))
    # Walk up until we hit an existing parent.
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        return Result(
            "Disk space",
            "warn",
            f"couldn't read disk usage at {path}: {e}",
        )
    free_gb = usage.free / (1024**3)
    if free_gb < 1:
        return Result(
            "Disk space",
            "error",
            f"{free_gb:.1f} GiB free on {path} (low)",
            hint=(
                "ChromaDB and Ollama models can fill quickly. Free space "
                "or move EQ_BRAIN_DATA_DIR / OLLAMA_MODELS to a roomier "
                "volume."
            ),
        )
    if free_gb < 10:
        return Result(
            "Disk space",
            "warn",
            f"{free_gb:.1f} GiB free on {path}",
        )
    return Result("Disk space", "ok", f"{free_gb:.1f} GiB free on {path}")


def check_optional_api_keys() -> Result:
    """Informational — which cloud LLM fallbacks are configured."""
    keys = {
        "Anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "Google": os.environ.get("GOOGLE_API_KEY", ""),
    }
    present = [name for name, val in keys.items() if val]
    if not present:
        return Result(
            "Cloud API keys",
            "warn",
            "none configured (local-only mode)",
            hint=("Optional. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY in .env for cloud fallback."),
        )
    return Result("Cloud API keys", "ok", ", ".join(present) + " configured")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

ALL_CHECKS = [
    check_python_version,
    check_local_brain_importable,
    check_chromadb_importable,
    check_brain,
    check_command_center,
    check_ollama,
    check_default_model,
    check_vault_dir,
    check_disk_space,
    check_optional_api_keys,
]


_LEVEL_BADGE = {
    "ok": "[green]✓ ok[/green]",
    "warn": "[yellow]⚠ warn[/yellow]",
    "error": "[red]✗ error[/red]",
}


def run(verbose: bool = False) -> int:
    """Run every check and print a summary table. Returns process exit code."""
    console = Console()
    results = [c() for c in ALL_CHECKS]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for r in results:
        table.add_row(r.name, _LEVEL_BADGE.get(r.level, r.level), r.message)
    console.print(table)

    # Hints — always print for warn/error; verbose extends to ok with comments.
    needs_hints = [r for r in results if r.hint and (verbose or r.level != "ok")]
    if needs_hints:
        console.print("\n[bold]Hints[/bold]")
        for r in needs_hints:
            console.print(f"  [bold]{r.name}[/bold]: {r.hint}")

    blocking = [r for r in results if r.is_blocking]
    if blocking:
        console.print(
            f"\n[red]{len(blocking)} blocking issue{'s' if len(blocking) != 1 else ''}.[/red]"
        )
        return 1

    warns = [r for r in results if r.level == "warn"]
    if warns:
        console.print(
            f"\n[yellow]{len(warns)} warning{'s' if len(warns) != 1 else ''}, no blockers.[/yellow]"
        )
    else:
        console.print("\n[green]All checks passed.[/green]")
    return 0
