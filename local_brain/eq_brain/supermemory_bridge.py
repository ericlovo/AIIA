"""
Supermemory Bridge — Bidirectional cloud sync for AIIA.

Connects AIIA's local memory (JSON + ChromaDB) to the Supermemory cloud service.
Two directions:
  - PUSH: Local memories (decisions, lessons, patterns, sessions) sync to cloud backup
  - PULL: Search DefaultApp's SME domain knowledge (legal, financial, compliance)

Container naming (separate from tenant containers):
  aiia_decisions   — Architecture decisions, tech choices
  aiia_lessons     — Lessons learned
  aiia_patterns    — Code patterns, conventions
  aiia_sessions    — Session summaries
  aiia_personal    — Household/life context
  aiia_team        — Team preferences
  aiia_project     — Project state
  aiia_agents      — Agent-specific memories
  aiia_wip         — Work-in-progress
  aiia_meta        — Self-reflection

Design:
  - Lazy init: reads SUPERMEMORY_API_KEY on first use
  - Kill switch: SUPERMEMORY_ENABLED=false disables all calls
  - asyncio.to_thread() wraps sync SDK, asyncio.wait_for() adds timeout
  - Graceful fallback: no API key -> logs warning, returns empty, AIIA works normally
  - Deterministic custom_id for dedup (Supermemory upserts on custom_id)
"""

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.eq_brain.supermemory_bridge")

# AIIA memory category -> Supermemory container
CATEGORY_CONTAINER_MAP = {
    "decisions": "aiia_decisions",
    "lessons": "aiia_lessons",
    "patterns": "aiia_patterns",
    "sessions": "aiia_sessions",
    "personal": "aiia_personal",
    "team": "aiia_team",
    "project": "aiia_project",
    "agents": "aiia_agents",
    "wip": "aiia_wip",
    "meta": "aiia_meta",
}

# DefaultApp SME domain -> Supermemory container (existing tenant containers)
SME_DOMAIN_CONTAINERS = {
    "finance": "aiia_sme_default_finance",
    "pi_law": "aiia_sme_default_pi_law",
    "family_law": "aiia_sme_default_family_law",
    "family_law_mn": "aiia_sme_default_family_law_mn",
    "compliance": "aiia_sme_default_compliance",
    "patent_law": "aiia_sme_default_patent_law",
    "client_system": "aiia_sme_default_client_system",
    "general": "aiia_sme_default_general",
    "eq": "aiia_sme_default_eq",
    "estate_planning": "aiia_sme_default_estate_planning",
    "wealth_management": "aiia_sme_default_wealth_management",
}


def _make_custom_id(category: str, content: str) -> str:
    """Deterministic custom_id for dedup. Same content -> same ID."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"aiia_{category}_{content_hash}"


class SupermemoryBridge:
    """
    Bidirectional bridge between AIIA's local memory and Supermemory cloud.

    Push: sync_memory(), sync_session(), sync_bulk()
    Pull: search_sme(), search_aiia_memories()
    """

    def __init__(self, timeout: float = 8.0):
        self._client = None
        self._initialized = False
        self._enabled = True
        self._timeout = timeout

    def _init_client(self):
        """Lazy init — called on first use. Supermemory SDK removed (April 2026)."""
        if self._initialized:
            return

        self._enabled = False
        self._client = None
        logger.info("Supermemory SDK removed — bridge permanently disabled")
        self._initialized = True

    @property
    def available(self) -> bool:
        """Whether the bridge is ready to make calls."""
        self._init_client()
        return self._client is not None and self._enabled

    # ──────────────────────────────────────────────────────────
    # PUSH — Local -> Cloud
    # ──────────────────────────────────────────────────────────

    async def sync_memory(
        self,
        fact: str,
        category: str = "lessons",
        source: str = "session",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Push one memory to matching Supermemory container.

        Uses deterministic custom_id for dedup — safe to call multiple times
        with the same fact.
        """
        self._init_client()
        if not self._client:
            return {"synced": False, "reason": "bridge_unavailable"}

        container = CATEGORY_CONTAINER_MAP.get(category, "aiia_lessons")
        custom_id = _make_custom_id(category, fact)

        sm_metadata = {
            "source": source,
            "category": category,
            "origin": "aiia_local",
        }
        if metadata:
            # SDK only accepts str/float/bool values — flatten nested dicts
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    sm_metadata[k] = v
                else:
                    sm_metadata[k] = str(v)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.add,
                    content=fact,
                    container_tag=container,
                    custom_id=custom_id,
                    metadata=sm_metadata,
                ),
                timeout=self._timeout,
            )
            logger.info(
                f"Synced to Supermemory [{container}] id={custom_id}: {fact[:60]}..."
            )
            return {"synced": True, "container": container, "custom_id": custom_id}
        except asyncio.TimeoutError:
            logger.warning(
                f"Supermemory sync timed out after {self._timeout}s for {custom_id}"
            )
            return {"synced": False, "reason": "timeout"}
        except Exception as e:
            logger.error(f"Supermemory sync failed: {e}")
            return {"synced": False, "reason": str(e)}

    async def sync_session(
        self,
        session_id: str,
        summary: str,
        decisions: Optional[List[str]] = None,
        lessons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Push a full session to cloud: summary + individual decisions/lessons.
        """
        results = {"session_synced": False, "decisions_synced": 0, "lessons_synced": 0}

        # Sync summary
        r = await self.sync_memory(
            fact=summary[:500],
            category="sessions",
            source=f"session:{session_id}",
            metadata={"session_id": session_id},
        )
        results["session_synced"] = r.get("synced", False)

        # Sync each decision
        if decisions:
            for decision in decisions:
                r = await self.sync_memory(
                    fact=decision,
                    category="decisions",
                    source=f"session:{session_id}",
                    metadata={"session_id": session_id},
                )
                if r.get("synced"):
                    results["decisions_synced"] += 1

        # Sync each lesson
        if lessons:
            for lesson in lessons:
                r = await self.sync_memory(
                    fact=lesson,
                    category="lessons",
                    source=f"session:{session_id}",
                    metadata={"session_id": session_id},
                )
                if r.get("synced"):
                    results["lessons_synced"] += 1

        return results

    async def sync_bulk(
        self,
        memories: List[Dict[str, Any]],
        category: str = "lessons",
    ) -> Dict[str, Any]:
        """
        Batch backfill existing JSON memories to cloud.
        Safe to re-run — dedup via custom_id.

        Each memory dict should have at least: {"fact": "...", "source": "..."}
        """
        self._init_client()
        if not self._client:
            return {
                "synced": 0,
                "skipped": 0,
                "errors": 0,
                "reason": "bridge_unavailable",
            }

        synced = 0
        errors = 0

        for mem in memories:
            fact = mem.get("fact", "")
            if not fact:
                continue

            source = mem.get("source", "bulk_sync")
            metadata = mem.get("metadata", {})

            r = await self.sync_memory(
                fact=fact,
                category=category,
                source=source,
                metadata=metadata,
            )
            if r.get("synced"):
                synced += 1
            elif r.get("reason") != "bridge_unavailable":
                errors += 1

        return {"synced": synced, "total": len(memories), "errors": errors}

    # ──────────────────────────────────────────────────────────
    # PULL — Cloud -> AIIA
    # ──────────────────────────────────────────────────────────

    async def search_sme(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        tenant_id: str = "default",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search DefaultApp's SME containers for domain knowledge.

        Args:
            query: What to search for
            domains: Which SME domains to search (default: all)
            tenant_id: Tenant whose SME to search (default: default)
            limit: Max results per domain
        """
        self._init_client()
        if not self._client:
            return []

        # Determine which containers to search
        if domains:
            containers = []
            for d in domains:
                container = SME_DOMAIN_CONTAINERS.get(d)
                if container:
                    # Replace default with actual tenant_id if different
                    if tenant_id != "default":
                        container = container.replace("default", tenant_id)
                    containers.append((d, container))
        else:
            containers = [(d, c) for d, c in SME_DOMAIN_CONTAINERS.items()]

        all_results = []
        for domain, container in containers:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.search.memories,
                        q=query,
                        container_tag=container,
                        limit=limit,
                    ),
                    timeout=self._timeout,
                )

                results = getattr(response, "results", [])
                for r in results:
                    if isinstance(r, dict):
                        content = r.get("memory", r.get("content", str(r)))
                        score = r.get("similarity", r.get("score", 0.0))
                    else:
                        content = getattr(r, "memory", getattr(r, "content", str(r)))
                        score = getattr(r, "similarity", getattr(r, "score", 0.0))

                    all_results.append(
                        {
                            "content": content,
                            "score": score,
                            "domain": domain,
                            "container": container,
                        }
                    )
            except asyncio.TimeoutError:
                logger.warning(f"SME search timed out for {container}")
            except Exception as e:
                logger.debug(f"SME search failed for {container}: {e}")

        # Sort by score descending, return top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]

    async def search_aiia_memories(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search AIIA's own cloud-synced containers.

        Args:
            query: What to search for
            categories: Which AIIA categories to search (default: all)
            limit: Max results total
        """
        self._init_client()
        if not self._client:
            return []

        if categories:
            containers = [
                (cat, CATEGORY_CONTAINER_MAP[cat])
                for cat in categories
                if cat in CATEGORY_CONTAINER_MAP
            ]
        else:
            containers = list(CATEGORY_CONTAINER_MAP.items())

        all_results = []
        for category, container in containers:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.search.memories,
                        q=query,
                        container_tag=container,
                        limit=limit,
                    ),
                    timeout=self._timeout,
                )

                results = getattr(response, "results", [])
                for r in results:
                    if isinstance(r, dict):
                        content = r.get("memory", r.get("content", str(r)))
                        score = r.get("similarity", r.get("score", 0.0))
                    else:
                        content = getattr(r, "memory", getattr(r, "content", str(r)))
                        score = getattr(r, "similarity", getattr(r, "score", 0.0))

                    all_results.append(
                        {
                            "content": content,
                            "score": score,
                            "category": category,
                            "container": container,
                        }
                    )
            except asyncio.TimeoutError:
                logger.warning(f"AIIA memory search timed out for {container}")
            except Exception as e:
                logger.debug(f"AIIA memory search failed for {container}: {e}")

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]

    # ──────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Health check for the bridge."""
        self._init_client()
        return {
            "available": self._client is not None,
            "enabled": self._enabled,
            "api_key_set": bool(os.getenv("SUPERMEMORY_API_KEY")),
            "timeout": self._timeout,
            "aiia_containers": list(CATEGORY_CONTAINER_MAP.values()),
            "sme_containers": list(SME_DOMAIN_CONTAINERS.values()),
        }
