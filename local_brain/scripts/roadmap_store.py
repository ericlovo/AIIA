"""
Roadmap store — JSON-backed CRUD for product stories.

Storage: ~/.aiia/eq_data/roadmap/stories.json

Usage:
    from local_brain.scripts.roadmap_store import RoadmapStore
    store = RoadmapStore()
    store.create(title="Dark mode", product="default-app", priority="P1")
    stories = store.list(product="default-app", status="active")
"""

import builtins
import json
import logging
import os
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from threading import RLock

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
    "assignee",
    "workstream_id",
)

# Similarity threshold for dedup (0.0 - 1.0)
DEDUP_THRESHOLD = 0.7


def _serialized_mutation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._delivery_lock:
            return method(self, *args, **kwargs)

    return wrapped


class RoadmapStore:
    def __init__(self, data_dir: str | None = None):
        self._dir = Path(data_dir) if data_dir else DATA_DIR
        self._file = self._dir / "stories.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.load_error = False
        self._stories: list[dict] = self._load()
        self._delivery_lock = RLock()

    def _load(self) -> list[dict]:
        if self._file.exists():
            try:
                payload = json.loads(self._file.read_text())
                if not isinstance(payload, list) or any(
                    not isinstance(s, dict) or not isinstance(s.get("id"), str) for s in payload
                ):
                    raise ValueError("invalid stories")
                return payload
            except (ValueError, OSError):
                self.load_error = True
                return []
        return []

    def _save(self) -> None:
        self._file.write_text(json.dumps(self._stories, indent=2))

    def delivery_stories(self, product):
        with self._delivery_lock:
            return deepcopy(self.list(product=product))

    def update_delivery_links(
        self, story_id: str, product: str, assignment_ids, pr_numbers, expected_updated_at=None
    ):
        """Persist explicit operator links atomically, outside generic story updates."""
        with self._delivery_lock:
            story = self.get(story_id)
            if story is None or story.get("product") != product:
                raise KeyError("feature_not_found")
            if self.load_error:
                raise OSError("roadmap_unavailable")
            if expected_updated_at is not None and story.get("updated_at") != expected_updated_at:
                raise ValueError("delivery_links_changed_refresh_required")
            updated = {
                **story,
                "assignment_ids": list(assignment_ids),
                "pr_numbers": list(pr_numbers),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            stories = [updated if item is story else item for item in self._stories]
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self._dir, delete=False) as out:
                    temporary = out.name
                    json.dump(stories, out, indent=2)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, self._file)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            self._stories = stories
            return updated

    def list(
        self,
        product: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        results = self._stories
        if product:
            results = [s for s in results if s.get("product") == product]
        if status:
            results = [s for s in results if s.get("status") == status]
        if tags:
            results = [s for s in results if any(t in s.get("tags", []) for t in tags)]
        return results

    def get(self, story_id: str) -> dict | None:
        for s in self._stories:
            if s["id"] == story_id:
                return s
        return None

    def find_similar(
        self,
        title: str,
        threshold: float = DEDUP_THRESHOLD,
        exclude_statuses: builtins.list[str] | None = None,
    ) -> builtins.list[dict]:
        """Find stories with similar titles. Returns matches above threshold."""
        exclude = set(exclude_statuses or ["shipped", "cancelled"])
        title_lower = title.lower().strip()
        matches = []
        for s in self._stories:
            if s.get("status") in exclude:
                continue
            ratio = SequenceMatcher(None, title_lower, s.get("title", "").lower().strip()).ratio()
            if ratio >= threshold:
                matches.append({**s, "_similarity": round(ratio, 2)})
        matches.sort(key=lambda x: x["_similarity"], reverse=True)
        return matches

    @_serialized_mutation
    def create(
        self,
        title: str,
        product: str,
        priority: str = "P2",
        status: str = "backlog",
        description: str = "",
        source_session: str = "",
        source_type: str = "manual",
        tags: builtins.list[str] | None = None,
        client_impact: str = "",
        related_stories: builtins.list[str] | None = None,
        dedup: bool = True,
    ) -> dict:
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

    @_serialized_mutation
    def update(self, story_id: str, **fields) -> dict | None:
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

    @_serialized_mutation
    def delete(self, story_id: str) -> bool:
        before = len(self._stories)
        self._stories = [s for s in self._stories if s["id"] != story_id]
        if len(self._stories) < before:
            self._save()
            return True
        return False

    def backlog_summary(self) -> dict:
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
