"""
AIIAMemoryExecutor — A2A executors for AIIA memory operations.

Two modes:
- remember: writes a fact to AIIA's structured memory (and the knowledge
  store + Supermemory cloud + Obsidian vault, depending on what's wired up).
  Input message text is parsed as either a plain fact or a tagged form
  "category: fact text" so the protocol stays text-only at the wire level
  while still letting callers route memories to specific buckets.
- search: vector-searches AIIA's knowledge store for the input query and
  returns the top N matching documents formatted as a single text artifact.

Both modes use the same lazy AIIAGetter pattern as AIIAExecutor so the
expensive ChromaDB + embeddings init only fires on first call.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal, Optional

from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.schema import Message, TextPart

logger = logging.getLogger("aplora.a2a.aiia_memory_executor")

AIIAGetter = Callable[[], Awaitable[Any]]
MemoryMode = Literal["remember", "search"]

DEFAULT_REMEMBER_CATEGORY = "lessons"
DEFAULT_REMEMBER_SOURCE = "a2a"
DEFAULT_SEARCH_RESULTS = 5
SEARCH_TEXT_TRUNCATE = 500


class AIIAMemoryExecutor(AgentExecutor):
    """
    Wraps AIIA's memory surface as an A2A executor.

    Pick the mode at construction time so each registered agent has a
    single, predictable behavior — keeps Agent Cards honest about what
    the agent does instead of dispatching on message content.
    """

    def __init__(
        self,
        aiia_getter: AIIAGetter,
        mode: MemoryMode,
        *,
        default_category: str = DEFAULT_REMEMBER_CATEGORY,
        default_source: str = DEFAULT_REMEMBER_SOURCE,
        n_results: int = DEFAULT_SEARCH_RESULTS,
    ) -> None:
        if mode not in ("remember", "search"):
            raise ValueError(
                f"AIIAMemoryExecutor mode must be remember|search, got {mode}"
            )
        self._get_aiia = aiia_getter
        self._mode = mode
        self._default_category = default_category
        self._default_source = default_source
        self._n_results = n_results

    async def execute(self, message: Message) -> ExecutorResult:
        text = _join_text_parts(message)
        if not text.strip():
            raise ValueError(
                f"AIIAMemoryExecutor[{self._mode}] requires a non-empty text part"
            )

        aiia = await self._get_aiia()
        if aiia is None:
            raise RuntimeError("AIIA is not available on this host")

        if self._mode == "remember":
            return await self._do_remember(aiia, text)
        return await self._do_search(aiia, text)

    # ─── remember ────────────────────────────────────────────────────

    async def _do_remember(self, aiia: Any, text: str) -> ExecutorResult:
        category, fact = _parse_tagged_fact(text, default=self._default_category)

        logger.info(
            "AIIAMemoryExecutor.remember category=%s len=%d",
            category,
            len(fact),
        )
        entry = await aiia.remember(
            fact=fact,
            category=category,
            source=self._default_source,
        )

        confirmation = f'AIIA remembered: "{fact}" [category: {category}]'
        metadata = {"category": category, "source": self._default_source}
        if isinstance(entry, dict):
            for key in ("id", "created_at", "timestamp"):
                if key in entry and _is_jsonable(entry[key]):
                    metadata[key] = entry[key]

        return ExecutorResult(
            parts=[TextPart(text=confirmation)],
            artifact_name="memory-entry",
            artifact_description=f"AIIA memory written to category '{category}'",
            metadata=metadata,
        )

    # ─── search ──────────────────────────────────────────────────────

    async def _do_search(self, aiia: Any, query: str) -> ExecutorResult:
        # AIIA exposes its vector store internally as `_knowledge` — same
        # access pattern AIIA uses on itself in eq_brain.brain._gather_context.
        # Single underscore is convention-private, not enforced; keeping the
        # executor in this module means we travel together if it ever changes.
        knowledge = getattr(aiia, "_knowledge", None)
        if knowledge is None or not hasattr(knowledge, "search"):
            raise RuntimeError("AIIA has no knowledge store available for search")

        logger.info(
            "AIIAMemoryExecutor.search n=%d query=%.80s", self._n_results, query
        )
        results = await knowledge.search(query, n_results=self._n_results)

        if not results:
            return ExecutorResult(
                parts=[TextPart(text="No matching documents found.")],
                artifact_name="search-results",
                artifact_description="Empty result set from AIIA knowledge search",
                metadata={"query": query, "n_results": 0},
            )

        formatted = _format_search_results(results)
        return ExecutorResult(
            parts=[TextPart(text=formatted)],
            artifact_name="search-results",
            artifact_description=f"Top {len(results)} matches from AIIA knowledge store",
            metadata={"query": query, "n_results": len(results)},
        )


# ─── helpers ─────────────────────────────────────────────────────────


def _join_text_parts(message: Message) -> str:
    texts = []
    for part in message.parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _parse_tagged_fact(text: str, *, default: str) -> tuple[str, str]:
    """
    Accept either a plain fact or a 'category: fact text' tagged form.

    The protocol stays text-only at the wire layer (no structured params),
    so we let callers prefix with 'category:' to route into a specific
    bucket. Unknown prefixes fall through as part of the fact body.
    """
    valid_categories = {
        "decisions",
        "patterns",
        "lessons",
        "team_knowledge",
        "agents",
        "sessions",
        "wip",
    }
    if ":" in text:
        head, _, rest = text.partition(":")
        head_norm = head.strip().lower()
        if head_norm in valid_categories and rest.strip():
            return head_norm, rest.strip()
    return default, text.strip()


def _format_search_results(results: list) -> str:
    """Render a list of knowledge_store.search() result dicts as text."""
    lines = []
    for i, r in enumerate(results, 1):
        if not isinstance(r, dict):
            lines.append(f"[{i}] {str(r)[:SEARCH_TEXT_TRUNCATE]}")
            continue
        source = r.get("source", "unknown")
        text = (r.get("text") or "")[:SEARCH_TEXT_TRUNCATE]
        score = r.get("relevance_score", r.get("distance", "?"))
        lines.append(f"[{i}] {source} (score: {score})\n{text}")
    return "\n\n".join(lines)


def _is_jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))
