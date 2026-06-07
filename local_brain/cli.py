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


def _http_get(url: str, timeout: float = 5.0) -> dict | None:
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
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
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
def doctor(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show hints even for passing checks."
    ),
) -> None:
    """Run environment + service health diagnostics.

    Checks Python version, package install, Brain/CC/Ollama reachability,
    default-model presence, vault directory, disk space, and optional
    cloud API keys. Each failed check includes an actionable hint.
    """
    from local_brain.cli_doctor import run

    rc = run(verbose=verbose)
    if rc:
        raise typer.Exit(rc)


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
    # briefing_cli reads sys.argv directly; rebuild what it expects.
    import sys

    from local_brain.scripts import briefing_cli

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
        console.print(
            f"\n[bold]Pulling story {top.get('id', '?')[:8]}:[/bold] {top.get('title', '?')}\n"
        )
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
        console.print(
            f"[dim]{e.get('created_at', '?')}[/dim]  [{e.get('category', '?')}]  {e.get('text', '')}"
        )


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


@app.command(name="journal-watch")
def journal_watch(
    poll_seconds: float = typer.Option(5.0, "--poll", help="Seconds between iCloud-folder polls."),
    inbox: str | None = typer.Option(
        None,
        "--inbox",
        help=(
            "Override the iCloud watch directory. Default is "
            "~/Library/Mobile Documents/com~apple~CloudDocs/AIIA-Inbox. "
            "Useful when you sync via Dropbox or a different cloud."
        ),
    ),
) -> None:
    """Run the iCloud watcher — processes audio files dropped by an iOS Shortcut.

    Long-running. Designed to be supervised by launchd in production (see
    scripts/com.aplora.aiia.journal-watch.plist + install-journal-watcher.sh).
    Ctrl-C exits cleanly.

    The watcher polls the configured inbox folder. When a new audio file
    appears (m4a, mp3, wav, etc.), it transcribes via Groq Whisper, distills
    via the first configured chat provider, writes the markdown to your
    Obsidian vault under 00-Inbox/, and archives the source audio.
    """
    import asyncio
    import logging
    import os
    from pathlib import Path

    from local_brain.journal.watcher import watch_forever

    logging.basicConfig(
        level=os.getenv("AIIA_JOURNAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    inbox_path = Path(inbox).expanduser() if inbox else None
    try:
        asyncio.run(watch_forever(poll_interval_seconds=poll_seconds, inbox=inbox_path))
    except KeyboardInterrupt:
        console.print("[dim]journal watcher stopped[/dim]")


research_app = typer.Typer(
    name="research",
    help="Autonomous deep-research topics — create, run sessions, read synthesis.",
)
app.add_typer(research_app)


def _http_post(url: str, body: dict, timeout: float = 600.0) -> dict | None:
    """POST JSON to the Brain and return the parsed dict (or None if unreachable)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


@research_app.command("list")
def research_list() -> None:
    """List research topics, newest first."""
    data = _http_get(f"{BRAIN_URL}/v1/research/topics")
    if data is None:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1)
    if not data:
        console.print(
            "[dim]No research topics yet. Create one with[/dim] [cyan]aiia research new[/cyan]"
        )
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Runs", justify="right")
    table.add_column("Sources", justify="right")
    table.add_column("Gaps", justify="right")
    for t in data:
        table.add_row(
            t.get("id", "?"),
            t.get("title", "?"),
            t.get("status", "?"),
            str(t.get("run_count", 0)),
            str(len(t.get("sources_indexed", []))),
            str(len(t.get("gaps", []))),
        )
    console.print(table)


@research_app.command("new")
def research_new(
    title: str = typer.Argument(..., help="Short topic title."),
    question: str = typer.Option(..., "-q", "--question", help="The research question to pursue."),
    seed: list[str] = typer.Option(
        None, "-s", "--seed", help="Seed URL to fetch first (repeatable)."
    ),
) -> None:
    """Create a new research topic."""
    data = _http_post(
        f"{BRAIN_URL}/v1/research/topics",
        {"title": title, "question": question, "seeds": seed or []},
        timeout=10,
    )
    if data is None:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1)
    console.print(
        f"[green]✓[/green] Created topic [bold]{data.get('id')}[/bold] — {data.get('title')}\n"
        f"  Run a session with: [cyan]aiia research run {data.get('id')}[/cyan]"
    )


@research_app.command("show")
def research_show(
    topic_id: str = typer.Argument(..., help="Topic ID."),
) -> None:
    """Show a topic's synthesis, open gaps, and indexed sources."""
    data = _http_get(f"{BRAIN_URL}/v1/research/topics/{topic_id}/synthesis")
    if data is None:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1)

    console.print(f"[bold]{data.get('title', '?')}[/bold]  [dim]({topic_id})[/dim]")
    console.print(f"[dim]Question:[/dim] {data.get('question', '?')}")
    console.print(
        f"[dim]Runs:[/dim] {data.get('run_count', 0)}   "
        f"[dim]Sources:[/dim] {len(data.get('sources_indexed', []))}   "
        f"[dim]Status:[/dim] {data.get('status', '?')}\n"
    )

    synthesis = data.get("synthesis") or "[dim](no synthesis yet — run a session)[/dim]"
    console.print("[bold]Synthesis[/bold]")
    console.print(synthesis)

    gaps = data.get("gaps", [])
    if gaps:
        console.print("\n[bold]Open gaps[/bold]")
        for g in gaps:
            console.print(f"  • {g}")


@research_app.command("run")
def research_run(
    topic_id: str = typer.Argument(..., help="Topic ID."),
) -> None:
    """Run one research session (streams progress as the engine works)."""
    req = urllib.request.Request(
        f"{BRAIN_URL}/v1/research/topics/{topic_id}/run",
        data=b"",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    console.print(f"[dim]Running research session on {topic_id}…[/dim]\n")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                _print_research_event(event)
    except urllib.error.URLError:
        _service_unreachable("Brain API", BRAIN_URL)
        raise typer.Exit(1) from None

    console.print(
        f"\n[dim]Session complete. View results:[/dim] [cyan]aiia research show {topic_id}[/cyan]"
    )


def _print_research_event(event: dict) -> None:
    """Render a single SSE event from the research run stream."""
    etype = event.get("type")
    if etype == "action":
        action = event.get("action", {})
        name = action.get("action", "?") if isinstance(action, dict) else str(action)
        console.print(f"  [cyan]→[/cyan] {name}")
    elif etype == "result":
        result = str(event.get("result", ""))
        console.print(f"    [dim]{result[:160]}[/dim]")
    elif etype == "done":
        console.print(f"  [green]✓[/green] {event.get('answer', '')[:300]}")
    elif etype == "error":
        console.print(f"  [red]✗[/red] {event.get('message', 'error')}")


def main() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    app()


if __name__ == "__main__":
    main()
