"""
Memory Quality Loop — Local consolidation, dedup, and quality scoring.

Every quality cycle (default every 6 hours):
1. Run memory consolidation (dedup, merge related entries)
2. Score unpromoted memories using a local LLM
3. Promote high-quality entries to the ChromaDB knowledge store
4. Budget-gated: max N promotions per cycle

Operates purely locally — no cloud sync. Tracks promoted IDs in a local
state file rather than mutating Memory internals, which keeps the loop
non-invasive and replay-safe.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from local_brain.config import AutonomyConfig

logger = logging.getLogger("aiia.autonomy.memory_quality")

# Only these categories are eligible for quality promotion
PROMOTABLE_CATEGORIES = ["decisions", "patterns", "lessons"]

_DEFAULT_STATE_FILENAME = "memory_quality_promoted.json"


class MemoryQualityLoop:
    """
    Autonomous memory quality improvement cycle.

    Uses a local state file to track promoted IDs instead of mutating
    Memory entries directly — keeps this loop non-invasive.
    """

    def __init__(
        self,
        config: AutonomyConfig,
        memory: Any,
        knowledge_store: Any,
        ollama: Any,
        consolidator: Optional[Any] = None,
        notify_fn: Any = None,
        state_dir: Optional[str] = None,
    ):
        self.config = config
        self.memory = memory
        self.knowledge_store = knowledge_store
        self.ollama = ollama
        self.consolidator = consolidator
        self._notify = notify_fn

        # Resolve state directory — defaults to the memory data dir's parent
        if state_dir is None and memory is not None:
            mem_dir = getattr(memory, "_data_dir", None)
            if isinstance(mem_dir, str) and mem_dir:
                state_dir = os.path.dirname(mem_dir)
        self._state_path = (
            os.path.join(state_dir, _DEFAULT_STATE_FILENAME)
            if isinstance(state_dir, str) and state_dir
            else None
        )
        self._promoted_ids: Set[str] = self._load_state()

    @property
    def enabled(self) -> bool:
        return self.config.level == "phase2" and self.config.memory_quality_enabled

    def _load_state(self) -> Set[str]:
        """Load previously-promoted memory IDs from disk."""
        if not self._state_path or not os.path.exists(self._state_path):
            return set()
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            return set(data.get("promoted_ids", []))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not load memory quality state: {e}")
            return set()

    def _save_state(self) -> None:
        """Persist promoted memory IDs to disk."""
        if not self._state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump({"promoted_ids": sorted(self._promoted_ids)}, f, indent=2)
        except OSError as e:
            logger.warning(f"Could not save memory quality state: {e}")

    async def run_quality_cycle(self) -> Dict[str, Any]:
        """
        Phase 1: Consolidation (dedup, merge related entries)
        Phase 2: Score unpromoted memories by quality
        Phase 3: Promote top entries to ChromaDB knowledge store
        """
        if not self.enabled:
            return {"skipped": True, "reason": "memory_quality_disabled"}

        results = {
            "consolidated": 0,
            "scored": 0,
            "promoted": 0,
            "errors": 0,
        }

        # Phase 1: Consolidation
        if self.consolidator:
            try:
                consolidation = await self.consolidator.run_all(
                    categories=PROMOTABLE_CATEGORIES,
                    force=False,
                )
                results["consolidated"] = consolidation.get("total_analyzed", 0)
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
                results["errors"] += 1

        # Phase 2: Score unpromoted memories. Memory API: recall() returns
        # entries in most-recent-first order; there is no get_all().
        candidates: List[Dict[str, Any]] = []
        for category in PROMOTABLE_CATEGORIES:
            try:
                entries = self.memory.recall(category=category, limit=500)
            except Exception as e:
                logger.warning(f"Memory recall failed for {category}: {e}")
                results["errors"] += 1
                continue

            for entry in entries:
                entry_id = entry.get("id", "")
                if entry_id and entry_id in self._promoted_ids:
                    continue
                if "category" not in entry:
                    entry["category"] = category
                candidates.append(entry)

        scored: List[tuple] = []
        for entry in candidates[:200]:  # Cap scoring batch
            try:
                score = await self._score_memory(entry)
                if score >= self.config.memory_quality_threshold:
                    scored.append((entry, score))
            except Exception as e:
                logger.debug(f"Scoring failed for {entry.get('id', '?')}: {e}")
                results["errors"] += 1

        results["scored"] = len(scored)

        # Phase 3: Promote top entries (budget-gated)
        scored.sort(key=lambda x: x[1], reverse=True)
        promoted = 0

        for entry, score in scored[: self.config.memory_quality_max_promotions]:
            try:
                await self._promote_to_knowledge(entry, score)
                promoted += 1
            except Exception as e:
                logger.warning(f"Promotion failed for {entry.get('id', '?')}: {e}")
                results["errors"] += 1
                break  # Stop on first failure to avoid cascading errors

        results["promoted"] = promoted

        if promoted > 0:
            self._save_state()

        if self._notify:
            await self._notify("memory_quality_cycle", results)

        logger.info(
            f"Memory quality cycle: consolidated={results['consolidated']}, "
            f"scored={results['scored']}, promoted={results['promoted']}"
        )

        return results

    async def _score_memory(self, entry: Dict[str, Any]) -> float:
        """
        Score a memory entry 0.0-1.0 on quality using the local LLM.

        Evaluates: relevance, specificity, actionability, long-term value.
        """
        content = entry.get("fact", entry.get("content", ""))
        category = entry.get("category", "unknown")

        if len(content) < 30:
            return 0.0  # Too short to be useful

        try:
            response = await self.ollama.chat(
                model="gemma4:e4b",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Score this memory entry 0.0-1.0 on quality "
                            f"(relevance to software development, specificity, "
                            f"actionability, long-term value). "
                            f"Respond with ONLY a float number, nothing else.\n\n"
                            f"Category: {category}\n"
                            f"Content: {content[:500]}"
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=10,
                num_ctx=2048,
            )
            text = response.get("message", {}).get("content", "").strip()
            return float(text)
        except (ValueError, TypeError):
            return 0.0

    async def _promote_to_knowledge(self, entry: Dict[str, Any], score: float) -> None:
        """Promote a memory entry to the ChromaDB knowledge store."""
        content = entry.get("fact", entry.get("content", ""))
        category = entry.get("category", "unknown")
        entry_id = entry.get("id", "")

        await self.knowledge_store.add_document(
            text=f"[{category}] {content}",
            source=f"memory_quality_promotion:{entry_id}",
            doc_type="memory",
            metadata={
                "category": category,
                "quality_score": score,
                "promoted_by": "memory_quality_loop",
                "original_id": entry_id,
            },
        )

        if entry_id:
            self._promoted_ids.add(entry_id)

        logger.debug(f"Promoted memory to KB: {entry_id} (category={category}, score={score:.2f})")
