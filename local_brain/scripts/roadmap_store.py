"""
Roadmap store — JSON-backed CRUD for product stories.

Storage: ~/.aiia/eq_data/roadmap/stories.json

Usage:
    from local_brain.scripts.roadmap_store import RoadmapStore
    store = RoadmapStore()
    store.create(title="Dark mode", product="my-app", priority="P1")
    stories = store.list(product="my-app", status="active")
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("aiia.roadmap")

DATA_DIR = Path.home() / ".aiia" / "eq_data" / "roadmap"
STORIES_FILE = DATA_DIR / "stories.json"

VALID_STATUSES = ("backlog", "active", "in_progress", "shipped", "blocked", "cancelled")
VALID_PRIORITIES = ("P0", "P1", "P2", "P3")

# Extended fields beyond the core set
EXTENDED_FIELDS = (
    "title",
    "product",
    "priority",
    "status",
    "description",
    "source_session",
    "source_type",
    "tags",
    "client_impact",
    "related_stories",
    "priority_score",
    "priority_reasoning",
)

# Similarity threshold for dedup (0.0 - 1.0)
DEDUP_THRESHOLD = 0.7


class RoadmapStore:
    def __init__(self, data_dir: Optional[str] = None):
        self._dir = Path(data_dir) if data_dir else DATA_DIR
        self._file = self._dir / "stories.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._stories: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        self._file.write_text(json.dumps(self._stories, indent=2))

    def list(
        self,
        product: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        results = self._stories
        if product:
            results = [s for s in results if s.get("product") == product]
        if status:
            results = [s for s in results if s.get("status") == status]
        if tags:
            results = [s for s in results if any(t in s.get("tags", []) for t in tags)]
        return results

    def get(self, story_id: str) -> Optional[Dict]:
        for s in self._stories:
            if s["id"] == story_id:
                return s
        return None

    def find_similar(
        self,
        title: str,
        threshold: float = DEDUP_THRESHOLD,
        exclude_statuses: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Find stories with similar titles. Returns matches above threshold."""
        exclude = set(exclude_statuses or ["shipped", "cancelled"])
        title_lower = title.lower().strip()
        matches = []
        for s in self._stories:
            if s.get("status") in exclude:
                continue
            ratio = SequenceMatcher(
                None, title_lower, s.get("title", "").lower().strip()
            ).ratio()
            if ratio >= threshold:
                matches.append({**s, "_similarity": round(ratio, 2)})
        matches.sort(key=lambda x: x["_similarity"], reverse=True)
        return matches

    def create(
        self,
        title: str,
        product: str,
        priority: str = "P2",
        status: str = "backlog",
        description: str = "",
        source_session: str = "",
        source_type: str = "manual",
        tags: Optional[List[str]] = None,
        client_impact: str = "",
        related_stories: Optional[List[str]] = None,
        dedup: bool = True,
    ) -> Dict:
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority: {priority}. Use {VALID_PRIORITIES}")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}. Use {VALID_STATUSES}")

        # Dedup check — return existing if highly similar
        if dedup:
            similar = self.find_similar(title, threshold=0.85)
            if similar:
                existing = similar[0]
                logger.info(
                    f"Dedup: '{title}' matches existing '{existing['title']}' "
                    f"(similarity={existing['_similarity']})"
                )
                # Update description if new one is more detailed
                if len(description) > len(existing.get("description", "")):
                    self.update(existing["id"], description=description)
                return {**existing, "_deduped": True}

        now = datetime.now(timezone.utc).isoformat()
        story = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "product": product,
            "priority": priority,
            "status": status,
            "description": description,
            "source_session": source_session,
            "source_type": source_type,
            "tags": tags or [],
            "client_impact": client_impact,
            "related_stories": related_stories or [],
            "created_at": now,
            "updated_at": now,
        }
        self._stories.append(story)
        self._save()
        return story

    def update(self, story_id: str, **fields) -> Optional[Dict]:
        for s in self._stories:
            if s["id"] == story_id:
                if "priority" in fields and fields["priority"] not in VALID_PRIORITIES:
                    raise ValueError(f"Invalid priority: {fields['priority']}")
                if "status" in fields and fields["status"] not in VALID_STATUSES:
                    raise ValueError(f"Invalid status: {fields['status']}")
                for key in EXTENDED_FIELDS:
                    if key in fields:
                        s[key] = fields[key]
                s["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return s
        return None

    def delete(self, story_id: str) -> bool:
        before = len(self._stories)
        self._stories = [s for s in self._stories if s["id"] != story_id]
        if len(self._stories) < before:
            self._save()
            return True
        return False

    def backlog_summary(self) -> Dict:
        """Summary stats for prioritization views."""
        by_priority = {}
        by_product = {}
        by_status = {}
        for s in self._stories:
            p = s.get("priority", "P2")
            prod = s.get("product", "unknown")
            st = s.get("status", "backlog")
            by_priority[p] = by_priority.get(p, 0) + 1
            by_product[prod] = by_product.get(prod, 0) + 1
            by_status[st] = by_status.get(st, 0) + 1
        return {
            "total": len(self._stories),
            "by_priority": by_priority,
            "by_product": by_product,
            "by_status": by_status,
        }
