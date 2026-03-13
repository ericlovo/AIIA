from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_brain.command_center.action_queue import (
        ActionQueue,
    )


@dataclass(frozen=True)
class ChainDefinition:
    trigger_type: str
    next_type: str
    next_title_template: str
    auto_approve: bool
    inherit_files: bool


VERIFY_TYPES = {"verify_lint", "verify_test", "verify_security"}

CHAINS: list[ChainDefinition] = [
    ChainDefinition(
        trigger_type="lint_fix",
        next_type="verify_lint",
        next_title_template="Verify: {parent_title}",
        auto_approve=True,
        inherit_files=True,
    ),
    ChainDefinition(
        trigger_type="test_fix",
        next_type="verify_test",
        next_title_template="Verify tests: {parent_title}",
        auto_approve=True,
        inherit_files=True,
    ),
    ChainDefinition(
        trigger_type="security_fix",
        next_type="verify_security",
        next_title_template="Verify security: {parent_title}",
        auto_approve=True,
        inherit_files=True,
    ),
]

_CHAIN_INDEX: dict[str, ChainDefinition] = {c.trigger_type: c for c in CHAINS}


def get_chain(action_type: str) -> ChainDefinition | None:
    return _CHAIN_INDEX.get(action_type)


def apply_chain(
    action_queue: ActionQueue, completed_action: dict[str, Any]
) -> dict[str, Any] | None:
    action_type = completed_action.get("type", completed_action.get("action_type", ""))
    chain = get_chain(action_type)
    if chain is None:
        return None

    parent_title = completed_action.get("title", action_type)
    title = chain.next_title_template.format(parent_title=parent_title)

    files = (
        completed_action.get("files_affected", completed_action.get("files", []))
        if chain.inherit_files
        else []
    )

    severity = completed_action.get("severity", "info")

    chained = action_queue.create_action(
        action_type=chain.next_type,
        severity=severity,
        title=title,
        description=f"Auto-chained from {completed_action.get('id', '?')}",
        files_affected=files,
        source_task="execution_chain",
    )
    # Link to parent
    chained["parent_id"] = completed_action.get("id")
    action_queue.save()

    if chain.auto_approve:
        action_queue.approve(chained["id"])

    return chained
