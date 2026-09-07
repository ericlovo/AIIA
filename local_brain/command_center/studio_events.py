"""Privacy-bounded event projection for the Agent Studio world view."""

from typing import Any

AGENT_FIELDS = ("id", "name", "status", "updated_at")
ASSIGNMENT_FIELDS = (
    "id",
    "title",
    "agent_id",
    "status",
    "source_handoff_id",
    "updated_at",
    "started_at",
    "completed_at",
)
HANDOFF_FIELDS = (
    "id",
    "source_assignment_id",
    "target_assignment_id",
    "from_agent_id",
    "to_agent_id",
    "artifact_type",
    "status",
    "updated_at",
)


def project(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: item.get(field) for field in fields}


def studio_event(entity: str, event: str, item: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "agent": AGENT_FIELDS,
        "assignment": ASSIGNMENT_FIELDS,
        "handoff": HANDOFF_FIELDS,
    }[entity]
    return {"entity": entity, "event": event, "item": project(item, fields)}


def studio_snapshot(
    agents: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    handoffs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "agents": [project(item, AGENT_FIELDS) for item in agents],
        "assignments": [project(item, ASSIGNMENT_FIELDS) for item in assignments],
        "handoffs": [project(item, HANDOFF_FIELDS) for item in handoffs],
    }
