"""
AIIAStatusExecutor — returns AIIA's health, knowledge, and memory stats.

Unlike AIIAExecutor (which routes through the LLM), this executor calls
brain.status() directly — a fast, deterministic read with no LLM invocation.
Returns: identity, model, knowledge doc count, memory stats by category,
and Supermemory connection status.

Use for monitoring dashboards, health checks, and "is AIIA alive?" queries
from other agents or the team dashboard.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.schema import Message, TextPart

logger = logging.getLogger("aplora.a2a.aiia_status_executor")

AIIAGetter = Callable[[], Awaitable[Any]]


class AIIAStatusExecutor(AgentExecutor):
    """Returns AIIA status as formatted text — no LLM call, just metadata."""

    def __init__(self, aiia_getter: AIIAGetter) -> None:
        self._get_aiia = aiia_getter

    async def execute(self, message: Message) -> ExecutorResult:
        aiia = await self._get_aiia()
        if aiia is None:
            raise RuntimeError("AIIA is not available on this host")

        status = await aiia.status()

        # Format as readable text
        lines = [
            "AIIA Status: online",
            f"Identity: {status.get('identity', '?')} — {status.get('name', '?')}",
            f"Model: {status.get('model', '?')}",
            f"Deep model: {status.get('deep_model', '?')}",
            "",
        ]

        knowledge = status.get("knowledge", {})
        lines.append(
            f"Knowledge: {knowledge.get('knowledge_docs', 0):,} docs "
            f"({knowledge.get('session_docs', 0)} session docs)"
        )

        memory = status.get("memory", {})
        total = memory.get("total_memories", 0)
        by_cat = memory.get("by_category", {})
        lines.append(f"Memory: {total:,} total memories")
        if by_cat:
            cat_parts = [f"  {cat}: {count}" for cat, count in sorted(by_cat.items())]
            lines.extend(cat_parts)

        sm = status.get("supermemory", {})
        lines.append(f"\nSupermemory: {'connected' if sm.get('available') else 'not connected'}")

        text_output = "\n".join(lines)
        return ExecutorResult(
            parts=[TextPart(text=text_output)],
            artifact_name="aiia-status",
            artifact_description="AIIA health, knowledge, and memory statistics",
            metadata=status,
        )
