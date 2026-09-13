"""Explicit Slack idea capture with request authentication and local storage."""

import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from local_brain.command_center.memory_inbox import MemoryInbox

router = APIRouter()


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
        "outbound_messages": False,
    }


@router.get("/api/memory-inbox")
def list_ideas(project: str = "", query: str = "", offset: int = 0):
    if offset < 0 or offset > 1_000_000 or len(query) > 500:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    try:
        return inbox().list(project=project, query=query, offset=offset)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc


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
