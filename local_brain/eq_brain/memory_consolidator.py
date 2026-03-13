"""
Memory Consolidator — Deep reasoning over AIIA's accumulated memories.

Runs nightly at 3am using DeepSeek R1 (14b) to:
- Cluster memories into themes
- Detect contradictions between entries
- Flag outdated information
- Compress redundant memories into insights

This is a batch worker — it uses AIIA's deep model path, not the
interactive fast model. DeepSeek cold-loads (~9GB VRAM), runs consolidation,
then unloads via keep_alive timeout.

Output is stored as structured JSON in the `meta` memory category
and as a report in ~/.aiia/logs/consolidation/.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from local_brain.eq_brain.brain import AIIA
from local_brain.eq_brain.memory import Memory

logger = logging.getLogger("aiia.consolidator")

CONSOLIDATION_CATEGORIES = ["patterns", "lessons", "decisions", "sessions"]

CONSOLIDATION_PROMPT = """You are AIIA's memory consolidation engine. Analyze these memories and produce a structured consolidation report.

## Memories ({category}, {count} entries)

{memories}

## Instructions

Analyze ALL memories above and produce a JSON object with exactly these fields:

1. **insights**: Array of theme clusters. Each has "theme" (string), "memory_ids" (array of IDs grouped under this theme), and "summary" (1-2 sentence synthesis).
2. **contradictions**: Array of detected contradictions. Each has "memory_id_a", "memory_id_b", "description" of the conflict.
3. **outdated**: Array of memory IDs that appear stale or superseded by newer entries, with "memory_id" and "reason".
4. **stats**: Object with "total_analyzed" (int), "themes_found" (int), "contradictions_found" (int), "outdated_found" (int), "compression_ratio" (float: insights count / total analyzed).

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""


class MemoryConsolidator:
    """Reads category memories, sends to DeepSeek, produces structured insights."""

    def __init__(self, aiia: AIIA, memory: Memory, data_dir: str):
        self._aiia = aiia
        self._memory = memory
        self._state_dir = os.path.join(data_dir, "sync")
        self._state_path = os.path.join(self._state_dir, "consolidation_state.json")
        os.makedirs(self._state_dir, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"last_run": None, "category_counts": {}}

    def _save_state(self, state: Dict[str, Any]):
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _needs_consolidation(self, category: str) -> bool:
        """Only consolidate if 3+ new memories since last run."""
        state = self._load_state()
        prev_count = state.get("category_counts", {}).get(category, 0)
        current = self._memory._read_category(category)
        return len(current) - prev_count >= 3

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        lines = []
        for m in memories:
            mid = m.get("id", "?")
            fact = m.get("fact", "")
            created = m.get("created_at", "?")
            source = m.get("source", "?")
            lines.append(f"[{mid}] ({created}, source={source}): {fact}")
        return "\n".join(lines)

    @staticmethod
    def _sanitize_json(text: str) -> str:
        """Fix common LLM JSON mistakes before parsing.

        DeepSeek sometimes writes:
        - Math expressions like `5 / 47` instead of floats
        - JS-style comments like `// Approximately 0.106`
        - Trailing commas before closing braces/brackets
        """
        # Strip single-line JS comments (// ...)
        text = re.sub(r"\s*//[^\n]*", "", text)
        # Strip multi-line JS comments (/* ... */)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # Replace `number / number` with the actual float
        def _eval_division(match):
            try:
                return str(round(int(match.group(1)) / int(match.group(2)), 4))
            except (ValueError, ZeroDivisionError):
                return "0"

        text = re.sub(r"(\d+)\s*/\s*(\d+)", _eval_division, text)
        # Strip trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """3-level fallback JSON parser for LLM output.

        1. Direct parse
        2. Extract from markdown code fence
        3. Find first { to last } substring

        Each level applies sanitization to fix common LLM JSON mistakes.
        """
        # Level 1: direct parse
        for text in [raw, MemoryConsolidator._sanitize_json(raw)]:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                pass

        # Level 2: markdown fence
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
            for text in [content, MemoryConsolidator._sanitize_json(content)]:
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Level 3: substring extraction
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            content = raw[first_brace : last_brace + 1]
            for text in [content, MemoryConsolidator._sanitize_json(content)]:
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    pass

        return None

    async def consolidate_category(self, category: str) -> Dict[str, Any]:
        """Consolidate a single memory category using deep reasoning."""
        memories = self._memory._read_category(category)
        if not memories:
            return {"category": category, "skipped": True, "reason": "empty"}

        logger.info(f"Consolidating {category}: {len(memories)} memories")
        start = time.monotonic()

        formatted = self._format_memories(memories)
        prompt = CONSOLIDATION_PROMPT.format(
            category=category,
            count=len(memories),
            memories=formatted,
        )

        result = await self._aiia.ask(
            question=prompt,
            include_sessions=False,
            n_results=0,
            depth="deep",
        )

        raw_answer = result.get("answer", "")
        reasoning = result.get("reasoning", "")
        latency = result.get("latency_ms", 0)

        if reasoning:
            logger.debug(f"DeepSeek reasoning for {category} ({len(reasoning)} chars)")

        logger.info(
            f"DeepSeek response for {category}: {len(raw_answer)} chars answer, "
            f"{len(reasoning)} chars reasoning"
        )

        parsed = self._parse_json_response(raw_answer)
        if not parsed:
            logger.warning(
                f"Failed to parse consolidation output for {category} "
                f"({len(raw_answer)} chars). Tail: ...{raw_answer[-100:]}"
            )
            return {
                "category": category,
                "error": "json_parse_failed",
                "raw_answer_length": len(raw_answer),
                "raw_answer_head": raw_answer[:300],
                "raw_answer_tail": raw_answer[-300:],
                "latency_ms": latency,
            }

        # Store consolidation result as meta memory
        summary_fact = (
            f"Consolidation [{category}]: {parsed.get('stats', {}).get('themes_found', 0)} themes, "
            f"{parsed.get('stats', {}).get('contradictions_found', 0)} contradictions, "
            f"{parsed.get('stats', {}).get('outdated_found', 0)} outdated entries "
            f"({datetime.utcnow().strftime('%Y-%m-%d')})"
        )
        self._memory.remember(
            fact=summary_fact,
            category="meta",
            source="consolidation",
            metadata={
                "type": "consolidation",
                "target_category": category,
                "insights_count": len(parsed.get("insights", [])),
                "contradictions_count": len(parsed.get("contradictions", [])),
                "outdated_count": len(parsed.get("outdated", [])),
            },
        )

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            f"Consolidated {category}: {len(parsed.get('insights', []))} insights, "
            f"{len(parsed.get('contradictions', []))} contradictions in {elapsed:.0f}ms"
        )

        return {
            "category": category,
            "insights": parsed.get("insights", []),
            "contradictions": parsed.get("contradictions", []),
            "outdated": parsed.get("outdated", []),
            "stats": parsed.get("stats", {}),
            "latency_ms": round(elapsed, 1),
            "reasoning_length": len(reasoning),
        }

    async def run_all(
        self,
        categories: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Consolidate all target categories.

        Args:
            categories: Override which categories to consolidate
            force: Bypass _needs_consolidation check
        """
        targets = categories or CONSOLIDATION_CATEGORIES
        start = time.monotonic()
        results = {}
        errors = []

        for cat in targets:
            if cat not in Memory.CATEGORIES:
                logger.warning(f"Skipping unknown category: {cat}")
                continue

            if not force and not self._needs_consolidation(cat):
                logger.info(f"Skipping {cat}: not enough new memories")
                results[cat] = {
                    "category": cat,
                    "skipped": True,
                    "reason": "not_enough_new",
                }
                continue

            try:
                result = await self.consolidate_category(cat)
                results[cat] = result
            except Exception as e:
                logger.error(f"Consolidation failed for {cat}: {e}", exc_info=True)
                results[cat] = {"category": cat, "error": str(e)}
                errors.append(f"{cat}: {e}")

        # Update state with current memory counts
        state = self._load_state()
        state["last_run"] = datetime.utcnow().isoformat()
        for cat in targets:
            if cat in Memory.CATEGORIES:
                state["category_counts"][cat] = len(self._memory._read_category(cat))
        self._save_state(state)

        elapsed = (time.monotonic() - start) * 1000
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "categories": results,
            "total_latency_ms": round(elapsed, 1),
            "errors": errors,
        }
