"""Catch Mini launchd environment leaking into tests that assume a vanilla host.

The scheduled Test Runner inherits AIIA_AIRGAP, TypeSafe/news/signals opt-ins,
a dummy-or-real TYPESAFE_API_KEY, and Sanction control-plane credentials.
Scrubbing airgap without scrubbing Sanction unmasks hybrid fail-closed, which
is how test_typesafe_advisor.py failed only on that host (CI stays green).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_brain.tests.conftest import _HOST_ENV_ISOLATE

ROOT = Path(__file__).resolve().parents[2]
_PROBE = "AIIA_HOST_ENV_LEAK_PROBE"

# Plausible Mini launchd environment. AIRGAP_ALLOWED_EGRESS is exported on
# that host even though production reads the frozenset, not this variable.
MINI_LAUNCHD_ENV = {
    "AIIA_AIRGAP": "1",
    "AIRGAP_ALLOWED_EGRESS": "xai.realtime,news.fetch,typesafe.signals,typesafe.routing",
    "AIIA_TYPESAFE_ENABLED": "1",
    "TYPESAFE_API_KEY": "host-typesafe-key",
    "AIIA_NEWS_ENABLED": "1",
    "AIIA_SIGNALS_ENABLED": "1",
    "AIIA_DEFAULT_MODEL": "qwen3:8b",
    "SANCTION_API_URL": "https://sanction.example.invalid/api/v1",
    "SANCTION_API_KEY": "host-sanction-key",
    "SANCTION_WALLET_ID": "wal_host",
}

_MINI_KEYS_THAT_MUST_BE_SCRUBBED = (
    "AIIA_AIRGAP",
    "AIIA_DEFAULT_MODEL",
    "AIIA_NEWS_ENABLED",
    "AIIA_SIGNALS_ENABLED",
    "AIIA_TYPESAFE_ENABLED",
    "AIRGAP_ALLOWED_EGRESS",
    "SANCTION_API_KEY",
    "SANCTION_API_URL",
    "SANCTION_WALLET_ID",
    "TYPESAFE_API_KEY",
)


def test_isolate_list_covers_mini_launchd_keys():
    missing = [key for key in _MINI_KEYS_THAT_MUST_BE_SCRUBBED if key not in _HOST_ENV_ISOLATE]
    assert missing == []
    for key in _MINI_KEYS_THAT_MUST_BE_SCRUBBED:
        assert os.environ.get(key) is None


def test_typesafe_advisor_stays_hermetic_under_mini_launchd_env():
    """Re-run the modules that leaked on the Mini under that host's env.

    Nested pytest is the point: it is how a future isolate-list omission
    would show up on CI instead of only on the scheduled runner.
    """
    if os.environ.get(_PROBE):
        pytest.skip("already inside the Mini-env probe")
    env = {**os.environ, **MINI_LAUNCHD_ENV, _PROBE: "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "local_brain/tests/test_typesafe_advisor.py",
            "local_brain/tests/test_airgap.py",
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
