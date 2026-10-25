import json
from datetime import datetime, timezone

import pytest

from local_brain.command_center import daily_digest as digest
from local_brain.command_center.deployment_verification import verified_receipt

NOW = datetime(2026, 10, 25, 20, tzinfo=timezone.utc)
SHA = "a" * 40


def receipt_file(tmp_path, monkeypatch, **changes):
    receipt = dict(
        state="verified",
        environment="local-production",
        sha=SHA,
        verified_at=NOW.isoformat(),
        health_passed=True,
        browser_passed=True,
        evidence="/tmp/verification-report.json",
    )
    receipt.update(changes)
    path = tmp_path / "receipts.json"
    path.write_text(json.dumps({"version": 1, "repositories": {"owner/repo": receipt}}))
    monkeypatch.setenv("AIIA_DEPLOYMENT_RECEIPTS", str(path))
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "failed"},
        {"state": "pending"},
        {"environment": "preview"},
        {"sha": "abc"},
        {"health_passed": False},
        {"browser_passed": False},
        {"evidence": ""},
        {"verified_at": "2026-10-26T00:00:00Z"},
        {"verified_at": "2026-10-01T00:00:00Z"},
        {"verified_at": "2026-10-25T20:00:00"},
    ],
)
def test_unverified_receipt_is_not_accepted(tmp_path, monkeypatch, changes):
    receipt_file(tmp_path, monkeypatch, **changes)
    assert verified_receipt("owner/repo", NOW) is None


def test_verified_receipt_is_repository_specific(tmp_path, monkeypatch):
    receipt_file(tmp_path, monkeypatch)
    assert verified_receipt("owner/repo", NOW)["sha"] == SHA
    assert verified_receipt("another/repo", NOW) is None


def test_malformed_receipt_fails_closed(tmp_path, monkeypatch):
    path = receipt_file(tmp_path, monkeypatch)
    path.write_text("{broken")
    assert verified_receipt("owner/repo", NOW) is None


def test_collector_requires_merge_commit_ancestry(tmp_path, monkeypatch):
    receipt_file(tmp_path, monkeypatch)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(digest, "_github_slug", lambda path: "owner/repo")
    monkeypatch.setattr(
        digest,
        "_merged_pr_rows",
        lambda *args: [
            digest.PullSignal(1, "Included", sha="b" * 40, merged=True),
            digest.PullSignal(2, "Not deployed", sha="c" * 40, merged=True),
        ],
    )

    def git(path, *args):
        if args[0] == "merge-base":
            return "b" * 40 if args[1] == "b" * 40 else "d" * 40
        if args[0] == "rev-list":
            return "0"
        return SHA

    result = digest.collect_repo_evidence(
        "demo", mounts={"demo": tmp_path}, git_read=git, github_api=lambda path: [], now=NOW
    )
    assert result.verified_deployments == {1: "local-production"}


def test_studio_and_digest_separate_verified_and_unverified_merges():
    product = digest.DigestProduct(id="demo", name="Demo", mount="demo")
    evidence = digest.RepoEvidence(
        repo_id="demo",
        mounted=True,
        complete=True,
        open_prs=0,
        merged_prs=[digest.PullSignal(1, "Deployed"), digest.PullSignal(2, "Merged only")],
        verified_deployments={1: "local-production"},
    )
    line = digest.morning_product_line(product, evidence, review=1)
    assert "verified deployed #1" in line.note
    assert "merged #2; deployment not verified" in line.note
    text = digest.format_slack_digest(
        date="2026-10-25",
        agents=[],
        assignments=[],
        loops={},
        tasks=[],
        inbox_counts={},
        products=[product],
        customers=[],
        repo_evidence=[evidence],
    )
    assert "*Verified deployed*\n- *Demo*: #1; local-production" in text
    assert "*Merged — deployment not verified*\n- *Demo*: #2" in text


def test_merged_pr_uses_merge_commit_not_branch_head():
    rows = [
        {
            "number": 5,
            "title": "Squashed",
            "merged_at": NOW.isoformat(),
            "head": {"sha": "b" * 40},
            "merge_commit_sha": SHA,
        }
    ]
    pulls = digest._merged_pr_rows(lambda path: rows, "owner/repo", NOW)
    assert pulls[0].sha == SHA
