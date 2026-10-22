"""Synthetic deployment evidence: no credentials or remote writes required."""

import subprocess
from unittest.mock import patch

from local_brain.command_center import project_deployments as subject


def deployment(id=1, environment="Preview", day=1):
    return dict(
        id=id,
        environment=environment,
        sha="a" * 40,
        created_at=f"2026-10-{day:02}T00:00:00Z",
        updated_at=f"2026-10-{day:02}T00:00:00Z",
    )


def status(state="success", **kwargs):
    return dict(
        id=100,
        state=state,
        created_at="2026-10-03T00:00:00Z",
        updated_at="2026-10-03T00:00:00Z",
        **kwargs,
    )


def test_preview_is_not_production():
    with patch.object(
        subject.repos, "_github_api", side_effect=[[deployment()], [status()]]
    ) as api:
        result = subject.collect_deployments("owner/repo")
    assert result["status"] == "available"
    assert [r["environment"] for r in result["environments"]] == ["Preview"]
    assert result["coverage"] == "bounded_recent_history"
    assert "not independently probed" in result["notice"]
    assert all(c.kwargs["timeout"] == 4 for c in api.call_args_list)


def test_latest_failure_never_replaced_by_older_success():
    with patch.object(
        subject.repos,
        "_github_api",
        side_effect=[
            [deployment(1), deployment(2, day=2)],
            [status("failure")],
        ],
    ) as api:
        result = subject.collect_deployments("owner/repo")
    assert result["environments"][0]["deployment_id"] == 2
    assert result["environments"][0]["state"] == "failure"
    assert api.call_count == 2


def test_malformed_newest_does_not_fall_back():
    invalid = deployment(2, day=2)
    invalid["sha"] = "main"
    with patch.object(subject.repos, "_github_api", return_value=[invalid, deployment()]):
        result = subject.collect_deployments("owner/repo")
    assert result["status"] == "unavailable"
    assert result["environments"] == []


def test_remote_failure_and_timeout():
    for error in (RuntimeError("private details"), subprocess.TimeoutExpired("gh", 4)):
        with patch.object(subject.repos, "_github_api", side_effect=error):
            result = subject.collect_deployments("owner/repo")
        assert result["status"] == "unavailable"
        assert "private details" not in str(result)


def test_status_failure_preserves_unknown_and_sha():
    with patch.object(subject.repos, "_github_api", side_effect=[[deployment()], RuntimeError()]):
        result = subject.collect_deployments("owner/repo")
    assert result["status"] == "partial"
    assert result["environments"][0]["state"] == "unknown"
    assert result["environments"][0]["sha"] == "a" * 40


def test_invalid_urls_are_not_exposed():
    for url in ("javascript:alert(1)", "https://user:pass@example.com", "http://example.com"):
        with patch.object(
            subject.repos,
            "_github_api",
            side_effect=[[deployment()], [status(environment_url=url)]],
        ):
            result = subject.collect_deployments("owner/repo")
        assert result["status"] == "partial"
        assert result["environments"][0]["environment_url"] is None


def test_request_budget_and_truncation():
    rows = [deployment(i + 1, f"env-{i}") for i in range(30)]
    with patch.object(subject.repos, "_github_api", side_effect=[rows] + [[status()]] * 4) as api:
        result = subject.collect_deployments("owner/repo")
    assert api.call_count == 5
    assert len(result["environments"]) == 4
    assert result["truncated"] and result["has_more"]
    assert result["status"] == "partial"


def test_bad_origin_never_fetches():
    with patch.object(subject.repos, "_github_api") as api:
        for slug in (None, "../repo", "owner/..", "owner/repo?token=x"):
            assert subject.collect_deployments(slug)["status"] == "unavailable"
        api.assert_not_called()


def test_empty_status_does_not_assert_success():
    with patch.object(subject.repos, "_github_api", side_effect=[[deployment()], []]):
        result = subject.collect_deployments("owner/repo")
    assert result["environments"][0]["state"] == "unknown"
