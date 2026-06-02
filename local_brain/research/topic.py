"""
ResearchTopic — state and persistence for a research question.

Topics live at {data_dir}/research/{id}.json. They accumulate across sessions:
seeds get indexed, gaps get logged, synthesis grows. Status: active → complete.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResearchTopic:
    id: str
    title: str
    question: str
    status: str = "active"  # active, paused, complete
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    seeds: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    synthesis: str = ""
    sources_indexed: list[str] = field(default_factory=list)
    run_count: int = 0
    last_run: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchTopic":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class TopicStore:
    """JSON-backed persistent store for research topics."""

    def __init__(self, data_dir: str):
        self._dir = Path(data_dir) / "research"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, topic_id: str) -> Path:
        return self._dir / f"{topic_id}.json"

    def save(self, topic: ResearchTopic) -> None:
        self._path(topic.id).write_text(json.dumps(topic.to_dict(), indent=2))

    def load(self, topic_id: str) -> ResearchTopic | None:
        p = self._path(topic_id)
        if not p.exists():
            return None
        try:
            return ResearchTopic.from_dict(json.loads(p.read_text()))
        except Exception:
            return None

    def list_all(self) -> list[ResearchTopic]:
        topics = []
        for f in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                topics.append(ResearchTopic.from_dict(json.loads(f.read_text())))
            except Exception:
                continue
        return topics

    def create(self, title: str, question: str, seeds: list[str] | None = None) -> ResearchTopic:
        topic = ResearchTopic(
            id=str(uuid.uuid4())[:8],
            title=title,
            question=question,
            seeds=seeds or [],
        )
        self.save(topic)
        return topic
