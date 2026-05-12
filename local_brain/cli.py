"""
⌬ AIIA — unified command-line interface.

Thin client over the running Brain API (:8100) and Command Center (:8200).
The point: give you one command (`aiia`) you reach for daily, instead of
remembering script paths or going through Claude Code via MCP.

Subcommands:
    aiia                          # show wordmark + quick tips
    aiia next                     # surface top stories ranked by priority × freshness
    aiia ask "..."                # one-shot query with memory + project context
    aiia memory list|search|add   # memory CRUD
    aiia briefing [--fresh]       # morning briefing (existing briefing_cli wrapper)
    aiia status                   # health of Brain + Command Center

Design:
    - This module is a dispatcher. The actual work lives in local_brain/scripts/
      and the Brain API. Each subcommand is a thin wrapper that fails fast
      with a friendly error if Brain/CC aren't reachable.
    - No new memory/story/LLM code — everything is delegated. v0.5 ships the
      surface; v0.6+ extends behind it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from local_brain.__version__ import __version__

BRAIN_URL = "http://localhost:8100"
COMMAND_CENTER_URL = "http://localhost:8200"

app = typer.Typer(
    name="aiia",
    help="⌬ AIIA — the memory layer your AI tools should have.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ----------------------------------------------------------------------------
# Wordmark + version
# ----------------------------------------------------------------------------

WORDMARK = (
    "[bold blue]─── ⌬ AIIA ──────────────────────────────────[/bold blue]\n"
    "  AI Information Architecture\n"
    "  Persistent memory · local-first · MCP-ready\n"
    f"  v{__version__} · Built at Aplora AI · Apache-2.0\n"
    "[bold blue]────────────────────────────────────────────────[/bold blue]"
)


def print_wordmark() -> None:
    console.print(WORDMARK)


def _http_get(url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET a JSON endpoint. Returns parsed dict or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _service_unreachable(service: str, url: str) -> None:
    """Print a consistent friendly error when a local service is down."""
    console.print(
        f"[red]✗[/red] {service} unreachable at [dim]{url}[/dim]\n"
        f"  Is the Brain API running? Try: [cyan]brain status[/cyan]"
    )


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit."
    ),
) -> None:
    """⌬ AIIA — the memory layer your AI tools should have."""
    if version:
        print_wordmark()
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        print_wordmark()
        console.print(
            "\n  [dim]Quick start:[/dim]\n"
            "    [cyan]aiia next[/cyan]      surface top stories to work on\n"
            "    [cyan]aiia ask[/cyan]       one-shot query with full context\n"
            "    [cyan]aiia status[/cyan]    Brain + Command Center health\n"
            "    [cyan]aiia --help[/cyan]    full command list\n"
        )


@app.command()
def status() -> None:
    """Show health of Brain API + Command Center."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Service")
    table.add_column("URL", style="dim")
    table.add_column("Status")

    brain = _http_get(f"{BRAIN_URL}/health")
    cc = _http_get(f"{COMMAND_CENTER_URL}/health")

    table.add_row(
        "Brain API",
        BRAIN_URL,
        "[green]✓ up[/green]" if brain else "[red]✗ down[/red]",
    )
    table.add_row(
        "Command Center",
        COMMAND_CENTER_URL,
        "[green]✓ up[/green]" if cc else "[red]✗ down[/red]",
    )
    console.print(table)

    if brain and "version" in brain:
        console.print(f"\n  Brain version: [bold]{brain.get('version', '?')}[/bold]")


@app.command()
def briefing(
    fresh: bool = typer.Option(
        False, "--fresh", "-f", help="Trigger a new briefing instead of fetching latest."
    ),
) -> None:
    """Fetch (or generate) AIIA's morning briefing from Command Center."""
    # Delegate to the existing briefing_cli script's main() so the surface
    # stays consistent with the `briefing` shell alias users already have.
    from local_brain.scripts import briefing_cli

    # briefing_cli reads sys.argv directly; rebuild what it expects.
    import sys

    saved_argv = sys.argv
    try:
        sys.argv = ["briefing_cli"]
        if fresh:
            sys.argv.append("--fresh")
        rc = briefing_cli.main()
        if rc:
            raise typer.Exit(rc)
    finally:
        sys.argv = saved_argv


@app.command(name="next")
def next_command(
    pull: bool = typer.Option(
        False, "--pull", help="Pick the top story and load full context for a session."
    ),
    limit: int = typer.Option(5, "-n", help="How many stories to surface."),
) -> None:
    """Surface top stories ranked by priority × freshness."""
    data = _http_get(f"{COMMAND_CENTER_URL}/api/stories/next?limit={limit}")
    if data is None:
        _service_unreachable("Command Center", COMMAND_CENTER_URL)
        raise typer.Exit(1)

    stories = data.get("stories", [])
    if not stories:
        console.print("[dim]No stories surfaced.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Priority")
    table.add_column("Source", style="dim")
    table.add_column("Title")

    for s in stories[:limit]:
        table.add_row(
            s.get("id", "?")[:8],
            str(s.get("score", "?")),
            s.get("priority", "?"),
            s.get("source", "?"),
            s.get("title", "?"),
        )
    console.print(table)

    if pull:
        # Pull the top one with full body + memory hits + project context.
        top = stories[0]
        console.print(f"\n[bold]Pulling story {top.get('id', '?')[:8]}:[/bold] {top.get('title', '?')}\n")
        body = top.get("body") or top.get("description") or "[dim](no body)[/dim]"
        console.print(body)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    n_results: int = typer.Option(
        5, "-n", "--n-results", help="How many memory hits to include in context."
    ),
) -> None:
    """One-shot query with memory + project context auto-loaded."""
    payload = json.dumps({"question": question, "n_results": n_results}).encode()
    req = urllib.request.Request(
        f"{BRAIN_URL}/v1/aiia/ask",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1) from None
    except json.JSONDecodeError:
        console.print("[red]✗[/red] Brain returned invalid JSON")
        raise typer.Exit(1) from None

    answer = data.get("answer") or data.get("response") or ""
    console.print(answer)


memory_app = typer.Typer(name="memory", help="Memory operations — list, search, add.")
app.add_typer(memory_app)


@memory_app.command("list")
def memory_list(
    limit: int = typer.Option(10, "-n", help="How many recent entries to show."),
) -> None:
    """List recent memory entries."""
    data = _http_get(f"{BRAIN_URL}/api/memory/recent?limit={limit}")
    if data is None:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1)

    entries = data.get("entries", [])
    if not entries:
        console.print("[dim]No memory entries.[/dim]")
        return

    for e in entries[:limit]:
        console.print(f"[dim]{e.get('created_at', '?')}[/dim]  [{e.get('category', '?')}]  {e.get('text', '')}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    n_results: int = typer.Option(5, "-n", "--n-results", help="How many results to return."),
) -> None:
    """Search memory by keyword."""
    payload = json.dumps({"question": query, "n_results": n_results}).encode()
    req = urllib.request.Request(
        f"{BRAIN_URL}/v1/aiia/search",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1) from None

    results = data.get("results", [])
    if not results:
        console.print("[dim]No results.[/dim]")
        return

    for r in results[:n_results]:
        console.print(f"[dim]{r.get('source', '?')}[/dim]  {r.get('content', '')[:200]}")


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(..., help="Memory content to save."),
    category: str = typer.Option(
        "decisions",
        "-c",
        "--category",
        help="Memory category (decisions, lessons, patterns, sessions, etc.).",
    ),
) -> None:
    """Add a memory entry."""
    payload = json.dumps({"text": text, "category": category}).encode()
    req = urllib.request.Request(
        f"{BRAIN_URL}/api/memory/add",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1) from None

    if data.get("ok") or data.get("id"):
        console.print(f"[green]✓[/green] Saved to [bold]{category}[/bold]")
    else:
        console.print(f"[red]✗[/red] Failed: {data}")
        raise typer.Exit(1)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    app()


if __name__ == "__main__":
    main()
