"""Bulk dismissal: only what a person selected, all or nothing, never a verdict."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from local_brain.command_center import assignment_registry as module
from local_brain.command_center.assignment_registry import (
    AssignmentRegistry,
    BulkDismissRejected,
)
from local_brain.command_center.persistence import PersistenceError


def _settled(registry: AssignmentRegistry, count: int, *, trigger: str = "interval") -> list[dict]:
    items = []
    for index in range(count):
        if trigger == "interval":
            item, _ = registry.create_scheduled_assignment(
                agent_id="agt_ci",
                agent_name="CI Signal Officer",
                objective="Inspect CI.",
                schedule_key=f"w{index}",
                interval_minutes=30,
            )
        else:
            item = registry.create_assignment(title=f"t{index}", objective="o", agent_id="agt")
        registry.set_running(item["id"])
        registry.finish_assignment(item["id"], result=f"report {index}")
        items.append(item)
    return items


def _picks(items):
    return [(item["id"], item["review_version"]) for item in items]


def test_dismisses_every_selected_item_and_keeps_verdicts(tmp_path):
    registry = AssignmentRegistry(tmp_path / "a.json")
    items = _settled(registry, 3)
    registry.review_assignment(
        items[0]["id"], decision="rejected", expected_version=items[0]["review_version"]
    )
    picks = _picks(items)

    dismissed = registry.dismiss_assignments(picks, note="Stale pre-guard backlog")

    assert [d["id"] for d in dismissed] == [i["id"] for i in items]
    restored = AssignmentRegistry(tmp_path / "a.json")
    for item in items:
        saved = restored.get_assignment(item["id"])
        assert saved["dismissed_at"]
        assert saved["dismiss_note"] == "Stale pre-guard backlog"
    # Dismissal is not a verdict: nothing becomes accepted, a rejection survives.
    assert restored.get_assignment(items[0]["id"])["review_status"] == "rejected"
    assert all(restored.get_assignment(i["id"])["review_status"] != "accepted" for i in items)
    assert restored.pending_loop_reviews("agt_ci") == 0  # releases the loop's review cap


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda reg, items, picks: picks.append(("asg_missing", "v")), "assignment_not_found"),
        (lambda reg, items, picks: picks.append(picks[0]), "duplicate_assignment"),
        (
            lambda reg, items, picks: picks.__setitem__(1, (items[1]["id"], "stale-version")),
            "review_changed_refresh_required",
        ),
        (
            lambda reg, items, picks: picks.append(
                (
                    reg.create_assignment(title="q", objective="o", agent_id="a")["id"],
                    "v",
                )
            ),
            "assignment_not_settled",
        ),
    ],
)
def test_any_bad_selection_dismisses_nothing(tmp_path, mutate, code):
    registry = AssignmentRegistry(tmp_path / "a.json")
    items = _settled(registry, 3)
    picks = _picks(items)
    mutate(registry, items, picks)
    before = (tmp_path / "a.json").read_bytes()

    with pytest.raises(BulkDismissRejected) as caught:
        registry.dismiss_assignments(picks, note="Stale")

    assert caught.value.code == code
    assert caught.value.assignment_id  # names the item that stopped the batch
    assert all(not registry.get_assignment(i["id"])["dismissed_at"] for i in items)
    assert (tmp_path / "a.json").read_bytes() == before


def test_already_dismissed_item_stops_the_batch(tmp_path):
    registry = AssignmentRegistry(tmp_path / "a.json")
    items = _settled(registry, 2)
    picks = _picks(items)
    registry.dismiss_assignment(items[1]["id"], dismissed=True, expected_version=picks[1][1])

    with pytest.raises(BulkDismissRejected) as caught:
        registry.dismiss_assignments(picks, note="Stale")

    assert caught.value.code == "assignment_already_dismissed"
    assert caught.value.assignment_id == items[1]["id"]
    assert not registry.get_assignment(items[0]["id"])["dismissed_at"]


@pytest.mark.parametrize(
    ("picks", "note", "code"),
    [([], "Stale", "no_assignments_selected"), ([("a", "v")], "   ", "dismiss_note_required")],
)
def test_requires_a_selection_and_a_reason(tmp_path, picks, note, code):
    registry = AssignmentRegistry(tmp_path / "a.json")
    with pytest.raises(BulkDismissRejected, match=code):
        registry.dismiss_assignments(picks, note=note)


def test_batch_size_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_BULK_DISMISS", 2)
    registry = AssignmentRegistry(tmp_path / "a.json")
    items = _settled(registry, 3)
    with pytest.raises(BulkDismissRejected, match="too_many_assignments"):
        registry.dismiss_assignments(_picks(items), note="Stale")


def test_failed_save_leaves_nothing_dismissed(tmp_path, monkeypatch):
    registry = AssignmentRegistry(tmp_path / "a.json")
    items = _settled(registry, 3)
    before = (tmp_path / "a.json").read_bytes()
    monkeypatch.setattr(module, "atomic_write_json", Mock(side_effect=PersistenceError("disk")))

    with pytest.raises(PersistenceError):
        registry.dismiss_assignments(_picks(items), note="Stale")

    assert all(not registry.get_assignment(i["id"])["dismissed_at"] for i in items)
    assert (tmp_path / "a.json").read_bytes() == before


async def test_endpoint_dismisses_and_reports_the_blocking_item(tmp_path, monkeypatch):
    from local_brain.command_center import server as cc

    registry = AssignmentRegistry(tmp_path / "a.json")
    monkeypatch.setattr(cc, "assignment_registry", registry)
    items = _settled(registry, 2, trigger="manual")
    body = cc.AssignmentBulkDismissRequest(
        items=[{"id": i["id"], "expected_version": i["review_version"]} for i in items],
        note="Stale",
    )

    result = await cc.dismiss_assignments(body)
    assert result["dismissed"] == 2

    with pytest.raises(HTTPException) as caught:
        await cc.dismiss_assignments(body)  # same versions are stale now
    assert caught.value.status_code == 409
    code, _, blocking = caught.value.detail.partition(":")
    assert code == "assignment_already_dismissed"
    assert blocking == items[0]["id"]


def test_request_model_rejects_empty_note_and_empty_selection():
    from pydantic import ValidationError

    from local_brain.command_center import server as cc

    with pytest.raises(ValidationError):
        cc.AssignmentBulkDismissRequest(items=[], note="x")
    with pytest.raises(ValidationError):
        cc.AssignmentBulkDismissRequest(items=[{"id": "a", "expected_version": "v"}], note="")
