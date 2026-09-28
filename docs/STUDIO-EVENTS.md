# Studio WebSocket contract

Verified against main at fa0fff4. Sources: command_center/studio_events.py,
server.py, assignment_registry.py and aiia_tasks.py. This documents existing
behavior; it introduces no protocol version, field additions or new egress.

## Envelope and connection snapshot

`/ws` sends JSON text with `{type: string, data: ...}`. On connection the server
sends `type: "init"`. Its `data.agent_studio` is the result of the Python helper
`studio_snapshot(agents, assignments, handoffs)`:

```json
{"agents": [], "assignments": [], "handoffs": []}
```

There is **no wire message named studio_snapshot**. The collections contain the
same projected records as live updates. Missing allowed fields become explicit
nulls; empty collections remain present. Array ordering is registry ordering,
not a protocol sorting guarantee.

The rest of init.data contains platform, summary, monitor, tasks, insights,
task_extra, routing, tokens, actions, action_summary, sessions, workstreams and
agent_world_layout. These are separate schemas, not protected by Studio's field
allowlists. task_extra contains code_health_trends, test_trends,
security_snapshot and security_trends. actions contains at most 20 pending
actions, not a complete action history. Use the action HTTP API for reconciliation.

## Projected live updates

```json
{
  "type": "agent_studio_update",
  "data": {
    "entity": "agent",
    "event": "updated",
    "item": {"id": "agent-id", "name": "Reviewer", "status": "idle", "updated_at": null}
  }
}
```

Current emitted entity/event pairs:

| Entity | Event names |
| --- | --- |
| agent | created, updated, deleted, running, failed, completed, skipped |
| assignment | created, updated, deleted, running, failed, completed, recovered |
| handoff | created, deleted, queued, running, failed, completed |

Handoff status events are dynamic: broadcast_assignment_event emits the
assignment first, then the linked handoff's current status when the handoff still
exists. Its queued/running/failed/completed values come from the registry, not a
separate event enum. Creation can emit both handoff.created and handoff.queued.
Event strings are not validated by the projection helper. Unknown entities raise
KeyError rather than falling back to unprojected data.

An event name is not necessarily the item's status: agent.completed can carry an
agent whose status is idle. Use item.status as state, event as change notification.
Deletion includes the projected former record; remove by entity plus item.id.
Agent IDs and assignment IDs belong to separate namespaces.

## Exact projection allowlists

All keys below are always emitted. Expected values are strings unless noted;
missing input values become null. The projector does not validate or coerce types.

| Entity | Fields |
| --- | --- |
| agent | id, name, status, updated_at |
| assignment | id, title, agent_id, status, source_handoff_id, updated_at, started_at, completed_at |
| handoff | id, source_assignment_id, target_assignment_id, from_agent_id, to_agent_id, artifact_type, status, updated_at |

Timestamps are record timestamps, normally ISO strings and sometimes null.
source_handoff_id can be an empty string. No mission, persona, tools, objective,
context, result, artifact, instructions, API key or token field is projected.
Tests hard-code these allowlists independently of the production constants.

This is **field minimization**, not content redaction: a secret embedded in a
permitted name/title would remain, and values are copied without recursive
sanitization. Do not treat it as an authorization or tenant-isolation boundary.

## Other event families on the same socket

These are not `{entity,event,item}` messages and must be dispatched separately:

| Wire type | Event / payload |
| --- | --- |
| session_update | registered, updated, closed; `{event, session: full session.to_dict()}` |
| agent_update | agent_changed; event, session_id, agent_name, agent_tier, agent_color, machine_id, current_task, chain_id, chain_position |
| workstream_update | created, updated, session_attached: `{event, workstream: full to_dict()}`; deleted: `{event, id}` |
| agent_world_layout | Layout registry snapshot, not an entity projection |
| action_updated | Action record; see ACTION-QUEUE.md |
| new_action | Task producer notification containing count and source; refetch actions |
| task_started, task_progress, task_complete, task_failed | Task lifecycle payloads from aiia_tasks.py |
| task_update | Full task list |
| new_insight | Insight record |
| monitor_update, routing_update, token_update | Respective service snapshots/statistics |
| interval_report | Session registry interval report |
| speech_done | Empty object |
| platform_update | Response to client `{type: "get_platform"}` |

Session heartbeat and workstream story attachment do not emit an update in the
current routes. Assignment review metadata is absent from Studio projection;
review/dismiss operations are not a reliable socket invalidation stream. Fetch
HTTP data after writes and on reconnect rather than assuming full event coverage.

## Consumer rules for Studio PR 5

- Seed projected collections from init.data.agent_studio, not from a fictional
  snapshot event. Keep detailed HTTP records separate from these partial records.
- Apply repeated updates by entity and ID; deletion is safe to repeat. Missing
  optional display values must tolerate null.
- Ignore unknown wire types/events safely; do not cast unrelated messages into
  Studio records. New schema additions need explicit review.
- Refetch after reconnect. There is no replay cursor, event ID, sequence number,
  acknowledgement or durable delivery. Initial snapshot and broadcasts are not
  an atomic subscription transaction; event order is not a consistency guarantee.
- Do not derive readiness, review state, spend or full output from these minimal
  records. Retrieve the corresponding HTTP resource when a detail view needs it.

## Verification

Projection tests assert exact event and snapshot shapes for all three entities,
explicit nulls, empty collections, input preservation and unknown-entity failure.
Fixtures include api_key, token and a secret-bearing extra object. A real
ConnectionManager serialization test checks the agent_studio_update envelope.
No frontend, runtime data or projection behavior was changed by this track.
