"""
Memory — Persistent structured memory for AIIA.

Unlike the vector knowledge store (which handles document search),
Memory stores structured facts, decisions, lessons, and team knowledge
as JSON. These are the things AIIA "knows" and can recall directly.

Categories:
    - decisions: Architecture decisions, tech choices, team agreements
    - patterns: Code patterns, conventions, anti-patterns to avoid
    - lessons: Things learned from debugging, incidents, failures
    - team: Team preferences, communication style, workflow
    - project: Project state, milestones, roadmap
    - agents: Agent-specific knowledge (per-agent memories)

Data lives at: {eq_data_dir}/memory/
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.eq_brain.memory")


class Memory:
    """
    Persistent structured memory for AIIA.

    Stores facts as JSON files organized by category.
    Fast reads (no LLM needed), durable across restarts.
    """

    CATEGORIES = [
        "decisions",  # Architecture decisions, tech choices
        "patterns",  # Code patterns and conventions
        "lessons",  # Things we learned the hard way
        "team",  # Team preferences and workflow
        "project",  # Project state and milestones
        "agents",  # Agent-specific memories
        "sessions",  # Session summaries
        "wip",  # Work-in-progress state for session continuity
        "meta",  # Self-reflection, memory digests
    ]

    def __init__(self, data_dir: str):
        self._data_dir = os.path.join(data_dir, "memory")
        os.makedirs(self._data_dir, exist_ok=True)
        self._version = 0  # Incremented on every write — used for prompt caching

        # Ensure category files exist
        for category in self.CATEGORIES:
            path = self._category_path(category)
            if not os.path.exists(path):
                self._write_category(category, [])

    def _category_path(self, category: str) -> str:
        return os.path.join(self._data_dir, f"{category}.json")

    def _read_category(self, category: str) -> List[Dict[str, Any]]:
        path = self._category_path(category)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_category(self, category: str, items: List[Dict[str, Any]]):
        path = self._category_path(category)
        with open(path, "w") as f:
            json.dump(items, f, indent=2, default=str)
        self._version += 1

    @property
    def version(self) -> int:
        """Monotonically increasing counter, bumped on every memory write."""
        return self._version

    def remember(
        self,
        fact: str,
        category: str = "lessons",
        source: str = "session",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a new fact in memory.

        Args:
            fact: The thing to remember (plain text)
            category: One of CATEGORIES
            source: Where this came from (session, bootstrap, user)
            metadata: Additional context
        """
        if category not in self.CATEGORIES:
            category = "lessons"

        items = self._read_category(category)

        # Check for duplicates (exact match)
        for item in items:
            if item.get("fact") == fact:
                logger.debug(f"Duplicate memory skipped: {fact[:50]}...")
                return item

        entry = {
            "id": f"{category}_{len(items)}_{int(time.time())}",
            "fact": fact,
            "source": source,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metadata": metadata or {},
        }

        items.append(entry)
        self._write_category(category, items)
        logger.info(f"Remembered [{category}]: {fact[:80]}...")

        return entry

    def recall(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Recall memories, optionally filtered by category.

        Returns most recent first.
        """
        if category and category in self.CATEGORIES:
            items = self._read_category(category)
            return list(reversed(items[-limit:]))

        # All categories
        all_items = []
        for cat in self.CATEGORIES:
            items = self._read_category(cat)
            for item in items:
                item["category"] = cat
            all_items.extend(items)

        # Sort by created_at descending
        all_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return all_items[:limit]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Simple keyword search across all memories.

        For semantic search, use KnowledgeStore instead.
        """
        query_lower = query.lower()
        matches = []

        for cat in self.CATEGORIES:
            items = self._read_category(cat)
            for item in items:
                fact = item.get("fact", "").lower()
                if query_lower in fact:
                    item["category"] = cat
                    matches.append(item)

        return matches[:limit]

    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory by ID."""
        for cat in self.CATEGORIES:
            items = self._read_category(cat)
            filtered = [i for i in items if i.get("id") != memory_id]
            if len(filtered) < len(items):
                self._write_category(cat, filtered)
                logger.info(f"Forgot memory: {memory_id}")
                return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        counts = {}
        total = 0
        for cat in self.CATEGORIES:
            items = self._read_category(cat)
            counts[cat] = len(items)
            total += len(items)

        return {
            "total_memories": total,
            "by_category": counts,
            "data_dir": self._data_dir,
        }

    def export_for_context(self, max_tokens: int = 4000) -> str:
        """
        Export key memories as a text block suitable for injection
        into an LLM system prompt. Prioritizes decisions and patterns.

        This is what gets sent to the brain agent as background knowledge.
        """
        lines = ["## What I Remember\n"]
        char_budget = max_tokens * 4  # Rough chars-to-tokens ratio

        # Priority order for context injection
        priority = ["decisions", "patterns", "lessons", "team", "project"]

        for cat in priority:
            items = self._read_category(cat)
            if not items:
                continue

            lines.append(f"\n### {cat.title()}")
            for item in items[-20:]:  # Most recent 20 per category
                fact = item.get("fact", "")
                lines.append(f"- {fact}")

            if sum(len(l) for l in lines) > char_budget:
                break

        return "\n".join(lines)
