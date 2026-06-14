"""Shared pytest fixtures + collection rules for the AIIA test suite.

The two jobs of this file:

1. **Skip tests blocked on dead imports.** A couple of test files still
   reference machinery that never landed on main: `test_autonomy_endpoint.py`
   needs an `autonomy_status()` API endpoint, and `test_phase2_config.py`
   needs `Gemma4Capabilities`. We collect-ignore those so the rest of the
   suite can run. (`test_autonomy.py` is no longer ignored — `AutonomyConfig`
   has been restored in `local_brain.config`.)

2. **Provide HTTP / service mocks** so tests can exercise CLI / API
   flows without a live Brain (`:8100`), Command Center (`:8200`), or
   Ollama (`:11434`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

# ----------------------------------------------------------------------------
# Collection rules
# ----------------------------------------------------------------------------
#
# Files in this list don't collect cleanly on `main` today. Each entry
# references a class that exists on an unmerged feature branch:
#
#   - test_autonomy.py             → needs AutonomyConfig (lost during
#                                    a refactor; the autonomy/* modules
#                                    themselves still import it)
#   - test_autonomy_endpoint.py    → same
#   - test_phase2_config.py        → needs Gemma4Capabilities (added on
#                                    the v0.5.0-dev.2-native-tools-opt-in
#                                    branch; never merged)
#
# Restoring these is its own focused PR. For now: skip collection so
# the rest of the suite (60 tests across 5 files) can run on every PR.
# Remove an entry the moment the underlying import is restored.

collect_ignore_glob = [
    # test_autonomy.py is restored: AutonomyConfig is back in local_brain.config.
    "test_autonomy_endpoint.py",  # still needs an autonomy_status() API endpoint
    "test_phase2_config.py",  # still needs Gemma4Capabilities
]


# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def fake_localhost_services(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set Brain / CC / Ollama URLs to localhost values for deterministic tests.

    Returns the env vars that were set so tests can re-derive the URLs.
    """
    urls = {
        "LOCAL_BRAIN_URL": "http://localhost:8100",
        "LOCAL_BRAIN_PORT": "8100",
        "COMMAND_CENTER_PORT": "8200",
        "OLLAMA_URL": "http://localhost:11434",
        "AIIA_DEFAULT_MODEL": "llama3.1:8b-instruct-q8_0",
    }
    for k, v in urls.items():
        monkeypatch.setenv(k, v)
    return urls


@pytest.fixture
def mock_http_get(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the urllib.request.urlopen pathway used by `aiia` CLI helpers.

    Tests configure the return value:

        def test_x(mock_http_get):
            mock_http_get.return_value = {"version": "0.5.0"}

    The mock is called as `urlopen(url, timeout=...).__enter__().read()`
    so we patch the deeper `urlopen` symbol used by both `cli._http_get`
    and `cli_doctor._http_get_status`.
    """
    import json
    import urllib.request

    payloads: dict[str, Any] = {"default": None}

    def fake_urlopen(url: str, timeout: float = 5.0) -> Any:
        payload = payloads.get(url, payloads.get("default"))

        class _Resp:
            status = 200

            def read(self) -> bytes:
                return json.dumps(payload or {}).encode()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    mock = MagicMock()
    mock.return_value = None  # acts like the parsed dict

    def configure(url: str, payload: Any) -> None:
        payloads[url] = payload

    mock.configure = configure  # type: ignore[attr-defined]
    return mock


@pytest.fixture
def isolated_aiia_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point EQ_BRAIN_DATA_DIR + HOME at a temp dir for the test.

    Use this when a test touches paths that default to ~/.aiia or
    EQ_BRAIN_DATA_DIR — keeps the developer's real vault untouched.
    """
    aiia_root = tmp_path / ".aiia"
    aiia_root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EQ_BRAIN_DATA_DIR", str(aiia_root / "eq_data"))
    yield str(aiia_root)


@pytest.fixture(autouse=True)
def _disable_real_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive: ensure tests never hit external telemetry endpoints.

    AIIA doesn't ship telemetry, but Chromadb has historically had
    optional anonymous-usage reporting. Lock it off so a CI run can't
    leak signal to a third party.
    """
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setenv("CHROMA_TELEMETRY_ENABLED", "0")
