"""Shared pytest fixtures + collection rules for the AIIA test suite.

The two jobs of this file:

1. **Skip tests blocked on dead imports.** Only `test_phase2_config.py`
   remains ignored — it needs `Gemma4Capabilities`, which never landed on
   main. (`test_autonomy.py` and `test_autonomy_endpoint.py` are no longer
   ignored: `AutonomyConfig` is back in `local_brain.config` and
   `autonomy_status()` now exists in `local_api`.)

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
# Files in this list don't collect cleanly on `main` today:
#
#   - test_phase2_config.py        → needs Gemma4Capabilities (added on
#                                    the v0.5.0-dev.2-native-tools-opt-in
#                                    branch; never merged)
#
# Remove an entry the moment the underlying import is restored.

collect_ignore_glob = [
    # test_autonomy.py + test_autonomy_endpoint.py are restored: AutonomyConfig
    # is back in local_brain.config and autonomy_status() exists in local_api.
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


# Launchd on the Mini exports production flags into the scheduled test
# runner. Tests that assert default-off / isolated behavior must not see
# those, and must not read or write the live ~/.aiia tree.
_HOST_ENV_ISOLATE = (
    "AIIA_AIRGAP",
    "AIIA_DEFAULT_MODEL",
    "AIIA_DEPLOYMENT_RECEIPTS",
    "AIIA_LIVE_TESTS",
    "AIIA_MEMORY_INBOX_PATH",
    "AIIA_NEWS_ENABLED",
    "AIIA_PRODUCTS_CONFIG",
    "AIIA_SIGNALS_ENABLED",
    "AIIA_SLACK_ACK_ENABLED",
    "AIIA_SLACK_BOT_TOKEN",
    "AIIA_SLACK_CHANNEL_IDS",
    "AIIA_SLACK_FILE_CAPTURE_ENABLED",
    "AIIA_SLACK_MEMORY_POST_CHANNEL_ID",
    "AIIA_SLACK_MEMORY_POST_ENABLED",
    "AIIA_SLACK_SIGNING_SECRET",
    "AIIA_SLACK_TEAM_ID",
    "AIIA_TEST_PYTHON",
    "AIIA_TYPESAFE_ENABLED",
    "COMMAND_CENTER_HOST",
    "LOCAL_ROUTING_MODEL",
    "LOCAL_TASK_MODEL",
    "OBSIDIAN_VAULT_DIR",
    "OLLAMA_TIMEOUT",
    "XAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_host_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the suite hermetic on a Mini that has real state and launchd env.

    The scheduled test runner inherits AIIA_AIRGAP, Slack/Typesafe opt-ins,
    the live default model, and HOME → ~/.aiia. Those are production
    settings, not test fixtures; leaking them made four assertions fail
    only on that host.
    """
    from local_brain import config as config_mod

    host = tmp_path_factory.mktemp("isolated-host")
    # Do not rewrite HOME: nested pytest/ruff subprocesses resolve user-site
    # packages from ~/.local. Point only AIIA data at a temp tree.
    monkeypatch.setenv("EQ_BRAIN_DATA_DIR", str(host / "eq_data"))
    for key in _HOST_ENV_ISOLATE:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_mod, "_config", None)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setenv("CHROMA_TELEMETRY_ENABLED", "0")
