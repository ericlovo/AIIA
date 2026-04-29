"""Smoke tests for the unified `aiia` CLI.

These don't require Brain or Command Center to be running — they exercise
the dispatcher, --help output, and graceful-error paths only. Tests that
require a live Brain belong in tests that mock the HTTP layer or that
run as integration tests against a real Mini.
"""

from __future__ import annotations

from typer.testing import CliRunner

from local_brain.__version__ import __version__
from local_brain.cli import app


runner = CliRunner()


class TestCliBasics:
    def test_no_args_shows_wordmark_and_quick_start(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "AIIA" in result.stdout
        assert "AI Information Architecture" in result.stdout
        assert "aiia next" in result.stdout
        assert "aiia ask" in result.stdout

    def test_version_flag_prints_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
        assert "AIIA" in result.stdout

    def test_help_lists_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("status", "briefing", "next", "ask", "memory"):
            assert cmd in result.stdout


class TestCliGracefulFailure:
    """When Brain/CC are down, the CLI should fail fast with friendly errors."""

    def test_status_when_services_down_does_not_crash(self):
        # status doesn't raise even when both services are unreachable —
        # it just shows "down" in the table.
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Brain API" in result.stdout
        assert "Command Center" in result.stdout

    def test_next_when_command_center_down_returns_nonzero(self):
        # No Command Center means no stories — exit non-zero with friendly message.
        result = runner.invoke(app, ["next"])
        # Exit 1 if the local CC is unreachable; if it happens to be up
        # during test, exit 0 is also acceptable.
        assert result.exit_code in (0, 1)
        if result.exit_code == 1:
            assert "unreachable" in result.stdout.lower()
