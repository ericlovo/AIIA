"""The Command Center listens on loopback unless explicitly overridden.

It has no auth of its own; the Cloudflare tunnel authenticates remote access at
the edge and reaches it on localhost. A 0.0.0.0 default would let the Mini's LAN
bypass that login, so the default is pinned here.
"""

from __future__ import annotations

import pytest


def test_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_brain.command_center import server

    monkeypatch.delenv("COMMAND_CENTER_HOST", raising=False)
    assert server.bind_host() == "127.0.0.1"


def test_env_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_brain.command_center import server

    monkeypatch.setenv("COMMAND_CENTER_HOST", "0.0.0.0")  # nosec B104
    assert server.bind_host() == "0.0.0.0"  # nosec B104
