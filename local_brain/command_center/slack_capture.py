"""Explicit Slack idea capture with request authentication and local storage."""

import asyncio
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from local_brain.command_center import slack_memory_posts, slack_receipts
from local_brain.command_center.memory_inbox import (
    IDEA_SORTS,
    IDEA_STATUSES,
    PRIORITIES,
    MemoryInbox,
)

BRAIN_URL = "http://localhost:8100"
BRAIN_TRANSPORT = None  # tests inject an httpx transport; production dials the local Brain
MEMORY_CATEGORIES = ("decisions", "patterns", "lessons", "project", "meta", "team", "agents")
MENTION = re.compile(r"<@[A-Z0-9]+>")


@asynccontextmanager
async def lifespan(app):
    tasks = [
        asyncio.create_task(slack_receipts.run_worker(inbox)),
        asyncio.create_task(slack_memory_posts.run_worker(inbox)),
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


router = APIRouter(lifespan=lifespan)


def inbox():
    return MemoryInbox(
        Path(
            os.getenv("AIIA_MEMORY_INBOX_PATH", str(Path(__file__).parent / "memory_inbox.sqlite3"))
        )
    )


def settings():
    return (
        os.getenv("AIIA_SLACK_SIGNING_SECRET", ""),
        os.getenv("AIIA_SLACK_TEAM_ID", ""),
        {
            item.strip()
            for item in os.getenv("AIIA_SLACK_CHANNEL_IDS", "").split(",")
            if item.strip()
        },
    )


@router.get("/api/integrations/slack/status")
def slack_status():
    secret, team, channels = settings()
    return {
        "configured": bool(secret and team and channels),
        "workspace_id": team,
        "channel_ids": sorted(channels),
        "mode": "explicit_idea_capture",
        "command": "/aiia-capture",
        "outbound_messages": slack_receipts.configured(),
        "acknowledgements_enabled": slack_receipts.enabled(),
        "acknowledgements_configured": slack_receipts.configured(),
        "acknowledgements": inbox().receipt_status() if inbox().path.exists() else {},
        "promotion_acknowledgements": inbox().receipt_status("promotion")
        if inbox().path.exists()
        else {},
        "memory_posts_enabled": slack_memory_posts.enabled(),
        "memory_posts_configured": slack_memory_posts.configured(),
        "memory_post_channel_id": slack_memory_posts.channel_id(),
        "memory_posts": inbox().memory_post_status() if inbox().path.exists() else {},
    }


@router.get("/api/memory-inbox")
def list_ideas(
    project: str = "",
    query: str = "",
    offset: int = 0,
    status: str = "",
    priority: str = "",
    sort: str = "newest",
):
    if offset < 0 or offset > 1_000_000 or len(query) > 500:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    if status and status not in IDEA_STATUSES:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    if (priority and priority not in PRIORITIES) or sort not in IDEA_SORTS:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    try:
        return inbox().list(
            project=project,
            query=query,
            offset=offset,
            status=status,
            priority=priority,
            sort=sort,
        )
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc


class PromoteRequest(BaseModel):
    category: str = "project"
    note: str = Field(default="", max_length=2_000)
    priority: str = "normal"
    post_to_slack: bool = False


class DismissRequest(BaseModel):
    note: str = Field(default="", max_length=2_000)


class MemoryRejected(Exception):
    pass


class MemoryUnavailable(Exception):
    pass


def capture_text(text: str) -> str:
    """The idea without the leading bot mention; the original text stays stored."""
    return MENTION.sub("", text).strip()


def memory_post_text(idea: dict, *, memory_id: str, category: str, priority: str) -> str:
    """The body approved at promote time and posted verbatim by the memory post worker."""
    return (
        f"[{priority.upper()}] Memory logged to {category}\n\n"
        f"{capture_text(idea['text'])}\n\n"
        f"Capture {idea['id'][:8]} · Memory {memory_id}"
    )


async def remember_in_brain(fact: str, category: str, metadata: dict) -> dict:
    """Store a reviewed capture as a Brain fact with provenance. Local call, no egress."""
    key = os.getenv("LOCAL_BRAIN_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=15, transport=BRAIN_TRANSPORT) as client:
            response = await client.post(
                f"{BRAIN_URL}/v1/aiia/remember",
                json={
                    "fact": fact,
                    "category": category,
                    "source": "slack:mindmoor",
                    "metadata": metadata,
                },
                headers={"x-api-key": key} if key else {},
            )
    except httpx.HTTPError as exc:
        raise MemoryUnavailable() from exc
    if response.status_code == 422:
        raise MemoryRejected()
    if response.status_code != 200:
        raise MemoryUnavailable()
    try:
        data = response.json()
    except ValueError as exc:
        raise MemoryUnavailable() from exc
    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]:
        raise MemoryUnavailable()
    return data


def _load_idea(idea_id: str) -> dict:
    try:
        idea = inbox().get(idea_id)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc
    if not idea:
        raise HTTPException(status_code=404, detail="idea_not_found")
    return idea


@router.post("/api/memory-inbox/{idea_id}/promote")
async def promote_idea(idea_id: str, body: PromoteRequest):
    if body.category not in MEMORY_CATEGORIES:
        raise HTTPException(status_code=422, detail="invalid_memory_category")
    if body.priority not in PRIORITIES:
        raise HTTPException(status_code=422, detail="invalid_priority")
    # Refuse before the Brain call so a disabled post never leaves a half-promoted capture.
    if body.post_to_slack and not slack_memory_posts.configured():
        raise HTTPException(status_code=409, detail="memory_posting_disabled")
    idea = _load_idea(idea_id)
    if idea["status"] == "promoted":
        raise HTTPException(status_code=409, detail="idea_already_promoted")
    fact = capture_text(idea["text"])
    if not fact:
        raise HTTPException(status_code=422, detail="idea_has_no_content")
    metadata = {
        "capture_id": idea["id"],
        "project": idea["project"],
        "source": idea["source"],
        "workspace_id": idea["workspace_id"],
        "channel_id": idea["channel_id"],
        "author_id": idea["author_id"],
        "captured_at": idea["created_at"],
    }
    if body.note.strip():
        metadata["review_note"] = body.note.strip()
    try:
        memory = await remember_in_brain(fact, body.category, metadata)
    except MemoryRejected as exc:
        raise HTTPException(status_code=422, detail="memory_quality_rejected") from exc
    except MemoryUnavailable as exc:
        raise HTTPException(status_code=503, detail="brain_unavailable") from exc
    post_channel_id = slack_memory_posts.channel_id() if body.post_to_slack else ""
    post_body = (
        memory_post_text(
            idea, memory_id=memory["id"], category=body.category, priority=body.priority
        )
        if body.post_to_slack
        else ""
    )
    try:
        updated = inbox().promote(
            idea_id,
            memory_id=memory["id"],
            category=body.category,
            note=body.note,
            priority=body.priority,
            post_channel_id=post_channel_id,
            post_body=post_body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, sqlite3.Error) as exc:
        # The Brain fact exists; the inbox row did not update. Say so instead of hiding it.
        raise HTTPException(status_code=503, detail="memory_saved_inbox_update_failed") from exc
    return {"idea": updated, "memory_id": memory["id"]}


@router.post("/api/memory-inbox/{idea_id}/dismiss")
def dismiss_idea(idea_id: str, body: DismissRequest):
    try:
        return {"idea": inbox().dismiss(idea_id, note=body.note)}
    except ValueError as exc:
        code = str(exc)
        raise HTTPException(
            status_code=404 if code == "idea_not_found" else 409, detail=code
        ) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc


@router.post("/api/memory-inbox/{idea_id}/restore")
def restore_idea(idea_id: str):
    try:
        return {"idea": inbox().restore(idea_id)}
    except ValueError as exc:
        code = str(exc)
        raise HTTPException(
            status_code=404 if code == "idea_not_found" else 409, detail=code
        ) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc


@router.post("/api/memory-inbox/{idea_id}/acknowledgement/retry")
def retry_acknowledgement(idea_id: str, kind: str = "capture"):
    if kind not in ("capture", "promotion", "memory_post"):
        raise HTTPException(status_code=422, detail="invalid_receipt_kind")
    if kind == "memory_post":
        if not slack_memory_posts.configured():
            raise HTTPException(status_code=409, detail="memory_posting_disabled")
    elif not slack_receipts.configured():
        raise HTTPException(status_code=503, detail="slack_receipts_not_configured")
    try:
        requeued = (
            inbox().retry_memory_post(idea_id)
            if kind == "memory_post"
            else inbox().retry_receipt(idea_id, kind)
        )
        if not requeued:
            raise HTTPException(status_code=409, detail="no_failed_receipt")
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc
    return {"status": "pending"}


async def verified_body(request: Request):
    secret, team, channels = settings()
    if not secret or not team or not channels:
        raise HTTPException(status_code=503, detail="slack_capture_not_configured")
    timestamp = request.headers.get("x-slack-request-timestamp", "")
    try:
        if abs(time.time() - int(timestamp)) > 300:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid_slack_signature") from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 32_768:
            raise HTTPException(status_code=413, detail="slack_request_too_large")
    expected = (
        "v0="
        + hmac.new(
            secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )
    if not hmac.compare_digest(
        expected.encode(), request.headers.get("x-slack-signature", "").encode()
    ):
        raise HTTPException(status_code=401, detail="invalid_slack_signature")
    return body, team, channels


@router.post("/api/integrations/slack/events")
async def capture_mention(request: Request):
    body, team, channels = await verified_body(request)
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError()
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_slack_payload") from exc
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge")
        if not isinstance(challenge, str) or not challenge or len(challenge) > 1024:
            raise HTTPException(status_code=400, detail="invalid_slack_payload")
        return {"challenge": challenge}
    if payload.get("team_id") != team:
        raise HTTPException(status_code=403, detail="slack_source_not_allowed")
    if payload.get("type") != "event_callback":
        return {"ok": True}
    event = payload.get("event")
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="invalid_slack_payload")
    if event.get("type") != "app_mention" or event.get("bot_id") or event.get("subtype"):
        return {"ok": True}
    channel = event.get("channel")
    if not isinstance(channel, str) or channel not in channels:
        raise HTTPException(status_code=403, detail="slack_source_not_allowed")
    values = [payload.get("event_id"), event.get("user"), event.get("text")]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise HTTPException(status_code=400, detail="invalid_slack_payload")
    event_id, author, text = values
    thread_ts = ""
    if slack_receipts.enabled():
        thread_ts = event.get("thread_ts") or event.get("ts")
        if not isinstance(thread_ts, str) or not re.fullmatch(
            r"[0-9]{1,16}\.[0-9]{1,6}", thread_ts
        ):
            raise HTTPException(status_code=400, detail="invalid_slack_timestamp")
    key = "slack:event:" + hashlib.sha256(f"{team}:{event_id}".encode()).hexdigest()
    try:
        inbox().capture(
            text=text,
            source_key=key,
            source="slack",
            project="mindmoor",
            workspace_id=team,
            channel_id=channel,
            author_id=author,
            receipt_thread_ts=thread_ts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_idea_length") from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc
    return {"ok": True}


@router.post("/api/integrations/slack/commands")
async def capture_command(request: Request):
    body, team, channels = await verified_body(request)
    try:
        form = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=30)
        required = ("team_id", "channel_id", "user_id", "command", "text", "trigger_id")
        if any(len(form.get(key, [])) != 1 for key in required):
            raise ValueError()
        fields = {key: form[key][0] for key in required}
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_slack_payload") from exc
    if fields["team_id"] != team or fields["channel_id"] not in channels:
        raise HTTPException(status_code=403, detail="slack_source_not_allowed")
    if not fields["user_id"] or not fields["trigger_id"]:
        raise HTTPException(status_code=400, detail="invalid_slack_payload")
    if fields["command"] != "/aiia-capture":
        raise HTTPException(status_code=400, detail="unsupported_slack_command")
    if not fields["text"].strip():
        return {"response_type": "ephemeral", "text": "Use /aiia-capture followed by your idea."}
    key = "slack:" + hashlib.sha256(f"{team}:{fields['trigger_id']}".encode()).hexdigest()
    try:
        idea = inbox().capture(
            text=fields["text"],
            source_key=key,
            source="slack",
            project="mindmoor",
            workspace_id=team,
            channel_id=fields["channel_id"],
            author_id=fields["user_id"],
        )
    except ValueError:
        return {"response_type": "ephemeral", "text": "Idea not saved: use 1 to 8,000 characters."}
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc
    return {
        "response_type": "ephemeral",
        "text": f"Saved idea {idea['id'][:8]} to the Performance Labs memory inbox for review.",
    }
