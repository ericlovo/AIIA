"""
Research API — /v1/research/* endpoints.

POST /v1/research/topics              — create a research topic
GET  /v1/research/topics              — list all topics
GET  /v1/research/topics/{id}         — get topic state
POST /v1/research/topics/{id}/run     — run one session (SSE stream)
GET  /v1/research/topics/{id}/synthesis — synthesis doc + gaps
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("aiia.research.router")

router = APIRouter(prefix="/v1/research", tags=["research"])

# Injected at startup via init_research()
_engine = None
_topic_store = None


def init_research(engine, topic_store) -> None:
    global _engine, _topic_store
    _engine = engine
    _topic_store = topic_store


def _require_store():
    if not _topic_store:
        raise HTTPException(status_code=503, detail="Research module not initialized")
    return _topic_store


def _require_engine():
    if not _engine:
        raise HTTPException(status_code=503, detail="Research engine not initialized")
    return _engine


class CreateTopicRequest(BaseModel):
    title: str
    question: str
    seeds: list[str] = []


@router.post("/topics", status_code=201)
async def create_topic(req: CreateTopicRequest):
    """Create a new research topic."""
    store = _require_store()
    topic = store.create(title=req.title, question=req.question, seeds=req.seeds)
    return topic.to_dict()


@router.get("/topics")
async def list_topics():
    """List all research topics, newest first."""
    store = _require_store()
    return [t.to_dict() for t in store.list_all()]


@router.get("/topics/{topic_id}")
async def get_topic(topic_id: str):
    """Get full topic state."""
    store = _require_store()
    topic = store.load(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")
    return topic.to_dict()


@router.post("/topics/{topic_id}/run")
async def run_research(topic_id: str):
    """Run one research session on a topic. Returns SSE stream of REPL events."""
    store = _require_store()
    engine = _require_engine()

    topic = store.load(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")
    if topic.status == "complete":
        raise HTTPException(
            status_code=409, detail="Topic is marked complete. Set status to 'active' to continue."
        )

    async def event_stream():
        try:
            async for event in engine.run(topic):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Research run failed for topic {topic_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/topics/{topic_id}/synthesis")
async def get_synthesis(topic_id: str):
    """Get the current synthesis document, gaps, and progress stats."""
    store = _require_store()
    topic = store.load(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")
    return {
        "topic_id": topic.id,
        "title": topic.title,
        "question": topic.question,
        "synthesis": topic.synthesis,
        "gaps": topic.gaps,
        "sources_indexed": topic.sources_indexed,
        "run_count": topic.run_count,
        "last_run": topic.last_run,
        "status": topic.status,
    }
