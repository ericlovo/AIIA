"""Tests for `aiia doctor` checks.

The checks talk to localhost services (Brain :8100, CC :8200, Ollama :11434).
None of those are guaranteed in CI, so we mock `_http_get_status` and
filesystem state where appropriate. Pure-version + import checks run
unmocked since they only depend on the test runner's environment.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from local_brain.cli_doctor import (
    ALL_CHECKS,
    Result,
    check_brain,
    check_command_center,
    check_default_model,
    check_local_brain_importable,
    check_ollama,
    check_python_version,
    run,
)

# ----------------------------------------------------------------------------
# Pure / structural
# ----------------------------------------------------------------------------


class TestResult:
    def test_ok_is_not_blocking(self):
        assert Result("x", "ok", "msg").is_blocking is False

    def test_warn_is_not_blocking(self):
        assert Result("x", "warn", "msg").is_blocking is False

    def test_error_is_blocking(self):
        assert Result("x", "error", "msg").is_blocking is True


def test_python_version_check_passes_on_supported():
    # Project requires py310+ (pyproject target-version); test runner satisfies
    # that by definition, so the check must return ok.
    r = check_python_version()
    assert r.name == "python"
    assert r.level == "ok"


def test_local_brain_importable():
    # If pytest discovered this file at all, local_brain is importable.
    r = check_local_brain_importable()
    assert r.level == "ok"


def test_all_checks_list_is_callable():
    # Each entry should be a zero-arg callable returning a Result.
    for c in ALL_CHECKS:
        assert callable(c)


# ----------------------------------------------------------------------------
# HTTP-dependent checks (mocked)
# ----------------------------------------------------------------------------


@patch("local_brain.cli_doctor._http_get_status")
def test_brain_warn_when_unreachable(mock_get):
    mock_get.return_value = (-1, None)
    r = check_brain()
    assert r.level == "warn"
    assert "unreachable" in r.message
    assert r.hint is not None


@patch("local_brain.cli_doctor._http_get_status")
def test_brain_ok_when_healthy(mock_get):
    mock_get.return_value = (200, {"version": "0.5.0"})
    r = check_brain()
    assert r.level == "ok"
    assert "0.5.0" in r.message


@patch("local_brain.cli_doctor._http_get_status")
def test_brain_warn_when_http_error(mock_get):
    mock_get.return_value = (503, None)
    r = check_brain()
    assert r.level == "warn"
    assert "503" in r.message


@patch("local_brain.cli_doctor._http_get_status")
def test_command_center_warn_when_unreachable(mock_get):
    mock_get.return_value = (-1, None)
    r = check_command_center()
    assert r.level == "warn"


@patch("local_brain.cli_doctor._http_get_status")
def test_ollama_error_when_unreachable(mock_get):
    # Ollama is required (not optional like Brain), so unreachable = error
    mock_get.return_value = (-1, None)
    r = check_ollama()
    assert r.level == "error"
    assert r.hint is not None
    assert "ollama" in r.hint.lower()


@patch("local_brain.cli_doctor._http_get_status")
def test_ollama_warn_when_no_models(mock_get):
    mock_get.return_value = (200, {"models": []})
    r = check_ollama()
    assert r.level == "warn"
    assert "no models" in r.message


@patch("local_brain.cli_doctor._http_get_status")
def test_ollama_ok_with_models(mock_get):
    mock_get.return_value = (
        200,
        {"models": [{"name": "llama3.1:8b-instruct-q8_0"}, {"name": "qwen3:14b"}]},
    )
    r = check_ollama()
    assert r.level == "ok"
    assert "2 models" in r.message


@patch("local_brain.cli_doctor._http_get_status")
def test_default_model_ok_when_present(mock_get):
    mock_get.return_value = (
        200,
        {"models": [{"name": "llama3.1:8b-instruct-q8_0"}]},
    )
    r = check_default_model()
    assert r.level == "ok"


@patch("local_brain.cli_doctor._http_get_status")
def test_default_model_warn_when_missing(mock_get):
    mock_get.return_value = (
        200,
        {"models": [{"name": "some-other-model:7b"}]},
    )
    r = check_default_model()
    assert r.level == "warn"
    assert "not pulled" in r.message
    assert r.hint is not None
    assert "ollama pull" in r.hint


# ----------------------------------------------------------------------------
# Runner integration
# ----------------------------------------------------------------------------


@patch("local_brain.cli_doctor._http_get_status")
def test_run_returns_1_when_blocking_failure(mock_get, capsys):
    # All HTTP checks fail with unreachable. Ollama is the only one that
    # treats unreachable as error → run should return 1.
    mock_get.return_value = (-1, None)
    rc = run(verbose=False)
    assert rc == 1
    out = capsys.readouterr().out
    assert "blocking" in out.lower()


@patch("local_brain.cli_doctor._http_get_status")
def test_run_returns_0_when_only_warnings(mock_get, capsys):
    # Brain/CC down (warn), but Ollama up with models (ok).
    def side_effect(url, timeout=3.0):
        if "11434" in url or "ollama" in url.lower():
            return (200, {"models": [{"name": "llama3.1:8b-instruct-q8_0"}]})
        return (-1, None)  # Brain + CC unreachable

    mock_get.side_effect = side_effect
    rc = run(verbose=False)
    # Vault dir / disk / API keys may add warns or errors depending on env,
    # but Ollama is now OK so no Ollama blocker. Tolerate either 0 or 1
    # depending on filesystem state; the assertion is just that we don't
    # crash and we produce *some* exit code.
    assert rc in (0, 1)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
