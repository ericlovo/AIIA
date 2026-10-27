"""Conservative attention identity and read-only legacy fixture discovery."""

import hashlib
import json


def evidence_key(item):
    if (
        item.get("status") != "completed"
        or not item.get("result", "").strip()
        or item.get("error")
        or item.get("review_status", "unreviewed") != "unreviewed"
        or (item.get("attempt_id") and not item.get("completed_run_id"))
    ):
        return ""
    fields = (
        "agent_id",
        "title",
        "objective",
        "context",
        "success_criteria",
        "priority",
        "source_kind",
        "source_ref",
        "source_handoff_id",
        "revision_of",
        "result",
    )
    # Manual run IDs identify occurrences, not the work's source scope.
    values = [
        "" if k == "source_ref" and item.get("source_kind") == "manual" else item.get(k, "")
        for k in fields
    ]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def acknowledged_keys(items):
    return {
        event["key"]
        for item in items
        for event in item.get("attention_events", [])
        if event.get("active")
    }


def fixture_preview(items, agent_ids, pending_ids):
    signatures = {
        "Inspect — Signal Officer": ("Inspect", "GREEN"),
        "Inspect again — Signal Officer": ("Inspect again", "GREEN"),
        "Inspect — A": ("Inspect", "OK"),
        "Inspect — Ops": ("Inspect", "OK"),
        "Inspect — Review": ("Inspect", "Observed evidence"),
        "Inspect — Reasoner": ("Inspect", "Observed evidence"),
        "Review changes — Review": ("Review changes", "Observed evidence"),
    }
    candidates, blocked = [], []
    for item in items:
        signature = signatures.get(item.get("title"))
        if not signature or signature != (item.get("objective"), item.get("result")):
            continue
        if (
            item.get("agent_id") in agent_ids
            or item.get("source_kind") != "manual"
            or any(
                item.get(k)
                for k in ("context", "success_criteria", "source_handoff_id", "revision_of")
            )
            or not evidence_key(item)
        ):
            continue
        row = {
            "id": item["id"],
            "title": item["title"],
            "result": item["result"],
            "review_version": item.get("review_version"),
            "fingerprint": evidence_key(item),
            "reason": "Exact test-mock task/output signature; absent agent; manual source; empty scope.",
        }
        (blocked if item["id"] in pending_ids else candidates).append(row)
    return {
        "candidates": candidates,
        "blocked": blocked,
        "preview_only": True,
        "warning": "Suspected test contamination, not confirmed provenance. Nothing has been classified or accepted.",
    }
