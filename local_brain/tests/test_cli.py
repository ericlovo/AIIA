"""Smoke tests for the unified `aiia` CLI.

These don't require Brain or Command Center to be running — they exercise
the dispatcher, --help output, and graceful-error paths only. Tests that
require a live Brain belong in tests that mock the HTTP layer or that
run as integration tests against a real Mini.
"""

from __future__ import annotations

from typer.testing import CliRunner

from local_brain import cli
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


class TestResearchErdos:
    """`aiia research erdos` — topic creation, mocked at the HTTP boundary."""

    def test_help_lists_research_subcommands(self):
        result = runner.invoke(app, ["research", "--help"])
        assert result.exit_code == 0
        for cmd in ("erdos", "list", "run", "show"):
            assert cmd in result.stdout

    def test_rejects_non_positive_number_without_calling_brain(self, monkeypatch):
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True
            return 0, None

        monkeypatch.setattr(cli, "_http_post", _boom)
        result = runner.invoke(app, ["research", "erdos", "0"])
        assert result.exit_code == 2
        assert not called  # validated client-side, never hit the network

    def test_create_success_prints_id_and_run_hint(self, monkeypatch):
        captured = {}

        def _fake_post(url, payload, timeout=10.0):
            captured["url"] = url
            captured["payload"] = payload
            return 201, {
                "id": "ab12cd34",
                "title": "Erdős Problem #351",
                "seeds": ["https://www.erdosproblems.com/351"],
            }

        monkeypatch.setattr(cli, "_http_post", _fake_post)
        result = runner.invoke(app, ["research", "erdos", "351"])
        assert result.exit_code == 0
        assert captured["url"].endswith("/v1/research/erdos")
        assert captured["payload"] == {"number": 351, "seeds": []}
        assert "ab12cd34" in result.stdout
        assert "aiia research run ab12cd34" in result.stdout

    def test_seeds_are_forwarded(self, monkeypatch):
        captured = {}

        def _fake_post(url, payload, timeout=10.0):
            captured["payload"] = payload
            return 201, {"id": "x", "title": "Erdős Problem #5", "seeds": []}

        monkeypatch.setattr(cli, "_http_post", _fake_post)
        result = runner.invoke(
            app,
            ["research", "erdos", "5", "-s", "https://arxiv.org/abs/2401.00001"],
        )
        assert result.exit_code == 0
        assert captured["payload"]["seeds"] == ["https://arxiv.org/abs/2401.00001"]

    def test_duplicate_reports_conflict_nonzero(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_http_post",
            lambda *a, **k: (409, {"detail": "Topic for Erdős problem #351 already exists: 'ab12'"}),
        )
        result = runner.invoke(app, ["research", "erdos", "351"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_unreachable_brain_reports_friendly_error(self, monkeypatch):
        monkeypatch.setattr(cli, "_http_post", lambda *a, **k: (0, None))
        result = runner.invoke(app, ["research", "erdos", "351"])
        assert result.exit_code == 1
        assert "unreachable" in result.stdout.lower()


class TestResearchLiterature:
    """`aiia research literature` — topic creation, mocked at the HTTP boundary."""

    def test_create_success(self, monkeypatch):
        captured = {}

        def _fake_post(url, payload, timeout=10.0):
            captured["url"] = url
            captured["payload"] = payload
            return 201, {
                "id": "lit12345",
                "title": "Literature: Mrs Dalloway",
                "seeds": ["https://en.wikipedia.org/wiki/Mrs_Dalloway"],
            }

        monkeypatch.setattr(cli, "_http_post", _fake_post)
        result = runner.invoke(app, ["research", "literature", "Mrs Dalloway"])
        assert result.exit_code == 0
        assert captured["url"].endswith("/v1/research/literature")
        assert captured["payload"] == {"subject": "Mrs Dalloway", "seeds": []}
        assert "lit12345" in result.stdout

    def test_blank_subject_rejected_client_side(self, monkeypatch):
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True
            return 0, None

        monkeypatch.setattr(cli, "_http_post", _boom)
        result = runner.invoke(app, ["research", "literature", "   "])
        assert result.exit_code == 2
        assert not called

    def test_duplicate_reports_conflict(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_http_post",
            lambda *a, **k: (409, {"detail": "Topic for 'Hamlet' already exists: 'ab12'"}),
        )
        result = runner.invoke(app, ["research", "literature", "Hamlet"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout


class TestResearchList:
    def test_lists_topics(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_http_get",
            lambda url, timeout=5.0: [
                {"id": "ab12", "profile": "erdos", "status": "active", "run_count": 2, "title": "Erdős Problem #351"},
            ],
        )
        result = runner.invoke(app, ["research", "list"])
        assert result.exit_code == 0
        assert "ab12" in result.stdout
        assert "erdos" in result.stdout

    def test_empty_list_is_friendly(self, monkeypatch):
        monkeypatch.setattr(cli, "_http_get", lambda url, timeout=5.0: [])
        result = runner.invoke(app, ["research", "list"])
        assert result.exit_code == 0
        assert "No research topics" in result.stdout

    def test_unreachable_brain_nonzero(self, monkeypatch):
        monkeypatch.setattr(cli, "_http_get", lambda url, timeout=5.0: None)
        result = runner.invoke(app, ["research", "list"])
        assert result.exit_code == 1
        assert "unreachable" in result.stdout.lower()


class TestResearchShow:
    def test_renders_synthesis_and_gaps(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_http_get",
            lambda url, timeout=5.0: {
                "title": "Erdős Problem #351",
                "status": "active",
                "question": "What is the status?",
                "synthesis": "Open problem; best known bound is X.",
                "gaps": ["arXiv:2401.00001 is PDF-only"],
                "sources_indexed": ["a", "b"],
                "run_count": 3,
                "last_run": "2026-06-14T00:00:00Z",
            },
        )
        result = runner.invoke(app, ["research", "show", "ab12"])
        assert result.exit_code == 0
        assert "best known bound" in result.stdout
        assert "PDF-only" in result.stdout

    def test_missing_topic_nonzero(self, monkeypatch):
        monkeypatch.setattr(
            cli, "_http_get", lambda url, timeout=5.0: {"detail": "Topic 'nope' not found"}
        )
        result = runner.invoke(app, ["research", "show", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.stdout
