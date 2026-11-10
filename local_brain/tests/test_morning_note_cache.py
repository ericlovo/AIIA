"""Studio morning-note: TTL cache, stale-while-revalidate, and source timeouts."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_brain.command_center import daily_digest
from local_brain.command_center.memory_inbox import MemoryInbox

PRODUCTS = [
    daily_digest.DigestProduct(
        id="aiia", name="AIIA", github="ericlovo/AIIA", mount="aiia", holds=(47, 52)
    ),
    daily_digest.DigestProduct(
        id="mindmoor",
        name="Mindmoor",
        github="tonybangert/mindmoor",
        mount="mindmoor",
        drift=("production", "alumni"),
    ),
]


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )


def _commit(path: Path, name: str, text: str) -> None:
    (path / name).write_text(text)
    _git(path, "add", name)
    _git(path, "commit", "-qm", text)


def _mounted_repo(root: Path, name: str, remote: str) -> Path:
    repo = root / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(repo, "remote", "add", "origin", remote)
    return repo


def _empty_digest() -> daily_digest.DigestResult:
    return daily_digest.DigestResult(
        line="CLEAR: no material drift/CI",
        severity="clear",
        fingerprint="test",
        evidence=[],
        customer_evidence=[],
    )


def _quiet_github(endpoint: str) -> object:
    if "/pulls?" in endpoint:
        return []
    if "/actions/runs" in endpoint:
        return {"workflow_runs": []}
    return {}


@pytest.fixture(autouse=True)
def _reset_morning_note_cache():
    from local_brain.command_center import server

    server.morning_note_cache.reset()
    yield
    server.morning_note_cache.reset()


def test_morning_note_cache_hit_skips_collect_digest(tmp_path, monkeypatch):
    from local_brain.command_center import server

    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    monkeypatch.setattr(server, "memory_capture_inbox", lambda: inbox)
    server.morning_note_cache.reset()
    calls: list[int] = []

    def collect(**kwargs):
        calls.append(1)
        return _empty_digest()

    monkeypatch.setattr(daily_digest, "collect_digest", collect)

    with TestClient(server.app) as client:
        first = client.get("/api/studio/morning-note")
        second = client.get("/api/studio/morning-note")

    assert first.status_code == 200
    assert second.status_code == 200
    body = first.json()
    assert body["generated_at"]
    assert body["generated_at"] == second.json()["generated_at"]
    assert "products" in body and "customers" in body and "decisions" in body
    assert "needs_you_total" in body
    assert len(calls) == 1


def test_morning_note_stale_refresh_returns_last_digest_immediately(tmp_path, monkeypatch):
    from local_brain.command_center import server

    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    monkeypatch.setattr(server, "memory_capture_inbox", lambda: inbox)
    server.morning_note_cache.reset()
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def collect(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _empty_digest()
        started.set()
        assert release.wait(2)
        return _empty_digest()

    monkeypatch.setattr(daily_digest, "collect_digest", collect)

    with TestClient(server.app) as client:
        first = client.get("/api/studio/morning-note")
        assert first.status_code == 200
        generated = first.json()["generated_at"]
        payload, status = server.morning_note_cache.lookup()
        assert status == "fresh"
        server.morning_note_cache.store(
            payload, now=time.monotonic() - server.morning_note_cache.ttl_seconds - 1
        )

        second = client.get("/api/studio/morning-note")
        assert second.status_code == 200
        assert second.json()["generated_at"] == generated
        assert started.wait(2)
        release.set()

        deadline = time.monotonic() + 2
        refreshed = None
        while time.monotonic() < deadline:
            refreshed, status = server.morning_note_cache.lookup()
            if refreshed and refreshed.get("generated_at") != generated:
                break
            time.sleep(0.02)
        assert refreshed is not None
        assert refreshed["generated_at"] != generated

        third = client.get("/api/studio/morning-note")
        assert third.status_code == 200
        assert third.json()["generated_at"] == refreshed["generated_at"]

    assert len(calls) == 2


def test_slow_source_returns_data_unavailable_without_blocking_others(tmp_path):
    fast = _mounted_repo(tmp_path, "aiia", "https://github.com/ericlovo/AIIA.git")
    slow = _mounted_repo(tmp_path, "mindmoor", "https://github.com/tonybangert/mindmoor.git")
    release = threading.Event()

    def github(endpoint: str) -> object:
        if "tonybangert/mindmoor" in endpoint:
            release.wait(30)
            return _quiet_github(endpoint)
        return _quiet_github(endpoint)

    started = time.monotonic()
    result = daily_digest.collect_digest(
        github_api=github,
        mounts={"aiia": fast, "mindmoor": slow},
        products=PRODUCTS,
        customers=[],
        source_timeout=0.3,
    )
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 2.0
    by_id = {row.repo_id: row for row in result.evidence}
    assert daily_digest.DATA_UNAVAILABLE in by_id["mindmoor"].failures
    assert by_id["mindmoor"].complete is False
    assert daily_digest.DATA_UNAVAILABLE not in by_id["aiia"].failures
    assert by_id["aiia"].open_prs == 0


def test_github_timeout_is_recorded_as_data_unavailable(tmp_path):
    repo = _mounted_repo(tmp_path, "aiia", "https://github.com/ericlovo/AIIA.git")

    def boom(endpoint: str) -> object:
        raise subprocess.TimeoutExpired("gh", 8)

    result = daily_digest.collect_digest(
        github_api=boom,
        mounts={"aiia": repo, "mindmoor": tmp_path / "missing"},
        products=PRODUCTS,
        customers=[],
        source_timeout=5.0,
    )
    aiia = next(row for row in result.evidence if row.repo_id == "aiia")
    assert daily_digest.DATA_UNAVAILABLE in aiia.failures
    assert result.severity == "incomplete"


def test_morning_note_cache_lookup_fresh_then_stale():
    cache = daily_digest.MorningNoteCache(ttl_seconds=30)
    assert cache.lookup(now=10) == (None, "miss")
    stored = cache.store({"date": "2026-11-10", "generated_at": "t0"}, now=10)
    payload, status = cache.lookup(now=20)
    assert status == "fresh"
    assert payload == stored
    payload, status = cache.lookup(now=50)
    assert status == "stale"
    assert payload["generated_at"] == "t0"
