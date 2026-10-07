"""Explicit Slack idea capture with request authentication and local storage."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from local_brain.command_center import slack_memory_posts, slack_receipts
from local_brain.command_center.memory_inbox import (
    IDEA_SORTS,
    IDEA_STATUSES,
    MAX_REVIEW_WINDOW_DAYS,
    PRIORITIES,
    REVIEW_OUTCOMES,
    UNCLASSIFIED,
    MemoryInbox,
)
from local_brain.egress import authorize_egress
from local_brain.research.fetcher import pdf_to_text

BRAIN_URL = "http://localhost:8100"
BRAIN_TRANSPORT = None  # tests inject an httpx transport; production dials the local Brain
SLACK_FILE_TRANSPORT = None  # tests inject an httpx transport; production dials Slack
MEMORY_CATEGORIES = ("decisions", "patterns", "lessons", "project", "meta", "team", "agents")
MENTION = re.compile(r"<@[A-Z0-9]+>")
MEMORY_POST_TEXT_LIMIT = 3_000
IDEA_CHAR_LIMIT = 8_000
FILE_CONTENT_CHAR_LIMIT = 3_000
FILE_DOWNLOAD_BYTE_LIMIT = 256_000
TRUNCATION_NOTE = "[truncated]"
TEXT_LIKE_MIMES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "application/json",
        "application/x-json",
    }
)
TEXT_LIKE_EXTS = frozenset({".md", ".txt", ".csv", ".json"})
TEXT_LIKE_FILETYPES = frozenset({"markdown", "text", "csv", "json"})
PDF_MIMES = frozenset({"application/pdf"})
logger = logging.getLogger(__name__)


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
    source: str = "",
    query: str = "",
    offset: int = 0,
    status: str = "",
    outcome: str = "",
    priority: str = "",
    sort: str = "newest",
):
    if offset < 0 or offset > 1_000_000 or len(query) > 500 or len(source) > 40:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    if outcome and outcome not in (*REVIEW_OUTCOMES, UNCLASSIFIED, "open"):
        raise HTTPException(status_code=422, detail="invalid_review_outcome")
    if status and status not in IDEA_STATUSES:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    if (priority and priority not in PRIORITIES) or sort not in IDEA_SORTS:
        raise HTTPException(status_code=422, detail="invalid_inbox_query")
    try:
        return inbox().list(
            project=project,
            source=source,
            query=query,
            outcome=outcome,
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


class TriageRequest(BaseModel):
    outcome: str
    note: str = Field(min_length=1, max_length=2_000)


class MemoryRejected(Exception):
    pass


class MemoryUnavailable(Exception):
    pass


def capture_text(text: str) -> str:
    """The idea without the leading bot mention; the original text stays stored."""
    return MENTION.sub("", text).strip()


def file_fetch_enabled() -> bool:
    return os.getenv("AIIA_SLACK_FILE_FETCH_ENABLED", "") == "1"


def event_files(event: dict) -> list[dict]:
    files = event.get("files")
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict)]


def file_header(file: dict) -> str:
    name = file.get("name") if isinstance(file.get("name"), str) and file["name"] else "unnamed"
    mimetype = (
        file.get("mimetype")
        if isinstance(file.get("mimetype"), str) and file["mimetype"]
        else "application/octet-stream"
    )
    size = file.get("size") if isinstance(file.get("size"), int) else 0
    permalink = file.get("permalink") if isinstance(file.get("permalink"), str) else ""
    return f"[file] {name} | {mimetype} | {size} bytes | {permalink}".rstrip()


def file_kind(file: dict) -> str:
    """Return 'pdf', 'text', or '' if the file should not be fetched."""
    name = str(file.get("name") or "").lower()
    mime = str(file.get("mimetype") or "").lower()
    filetype = str(file.get("filetype") or "").lower()
    mode = str(file.get("mode") or "").lower()
    ext = f".{name.rsplit('.', 1)[-1]}" if "." in name else ""
    if mime in PDF_MIMES or ext == ".pdf" or filetype == "pdf":
        return "pdf"
    if (
        mime in TEXT_LIKE_MIMES
        or ext in TEXT_LIKE_EXTS
        or filetype in TEXT_LIKE_FILETYPES
        or mode == "snippet"
    ):
        return "text"
    return ""


def slack_download_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname == "files.slack.com"


def extract_file_text(data: bytes, kind: str) -> str | None:
    if kind == "pdf":
        try:
            text = pdf_to_text(data)
        except Exception:
            return None
        return text or None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")


def build_captured_text(text: str, files: list[dict], contents: list[str | None]) -> str:
    parts: list[str] = []
    if text:
        parts.append(text)
    for file, content in zip(files, contents, strict=True):
        header = file_header(file)
        if content is None:
            parts.append(header)
            continue
        body = content
        truncated = False
        if len(body) > FILE_CONTENT_CHAR_LIMIT:
            body = body[:FILE_CONTENT_CHAR_LIMIT]
            truncated = True
        block = f"{header}\n{body}"
        if truncated:
            block += f"\n{TRUNCATION_NOTE}"
        parts.append(block)
    assembled = "\n\n".join(parts)
    if len(assembled) <= IDEA_CHAR_LIMIT:
        return assembled
    note = f"\n{TRUNCATION_NOTE}"
    return assembled[: IDEA_CHAR_LIMIT - len(note)] + note


async def fetch_one_file(client: httpx.AsyncClient, file: dict, headers: dict) -> str | None:
    kind = file_kind(file)
    if not kind:
        return None
    file_id = file.get("id")
    if not isinstance(file_id, str) or not file_id:
        return None
    try:
        info = await client.get(
            "https://slack.com/api/files.info",
            params={"file": file_id},
            headers=headers,
        )
        payload = info.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            return None
        meta = payload.get("file")
        if not isinstance(meta, dict):
            return None
        url = meta.get("url_private_download")
        if not isinstance(url, str) or not slack_download_allowed(url):
            return None
        downloaded = await client.get(url, headers=headers)
        if downloaded.status_code != 200:
            return None
        data = downloaded.content[:FILE_DOWNLOAD_BYTE_LIMIT]
        if not data:
            return None
        return extract_file_text(data, kind)
    except (httpx.HTTPError, ValueError, OSError):
        logger.warning("slack file fetch failed")
        return None


async def fetch_file_contents(files: list[dict]) -> list[str | None]:
    contents: list[str | None] = [None] * len(files)
    if not files or not file_fetch_enabled() or not any(file_kind(file) for file in files):
        return contents
    token = os.getenv("AIIA_SLACK_BOT_TOKEN", "")
    if not token:
        return contents
    decision = await authorize_egress("slack.file_fetch", server="files.slack.com")
    if not decision.allowed:
        return contents
    headers = {"Authorization": "Bearer " + token}
    try:
        async with httpx.AsyncClient(
            timeout=1.5,
            follow_redirects=False,
            transport=SLACK_FILE_TRANSPORT,
        ) as client:
            fetched = await asyncio.gather(
                *(fetch_one_file(client, file, headers) for file in files),
                return_exceptions=True,
            )
        for index, result in enumerate(fetched):
            if isinstance(result, str):
                contents[index] = result
            elif isinstance(result, BaseException):
                logger.warning("slack file fetch failed")
    except (httpx.HTTPError, ValueError, OSError):
        logger.warning("slack file fetch failed")
    return contents


def slack_escape(text: str) -> str:
    """Neutralize Slack control syntax: <!channel>, <!here>, <@U…> and <url|label> links."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slack_unescape(text: str) -> str:
    """Undo Slack's own &amp; &lt; &gt; encoding of inbound message text; &amp; goes last."""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def memory_post_text(idea: dict, *, memory_id: str, category: str, priority: str) -> str:
    """The body approved at promote time and posted verbatim by the memory post worker.

    Slack delivers captures with & < > already encoded, so that encoding is undone
    first or "R&D" would post as "R&amp;D"; the escape below still runs over every
    character, so a decoded <!channel> is re-neutralized. The text is capped before
    escaping so an entity is never cut in half; the whole body is escaped because
    the memory id comes back from the Brain.
    """
    text = capture_text(idea["text"])
    if idea.get("source") == "slack":
        text = slack_unescape(text)
    truncated = len(text) > MEMORY_POST_TEXT_LIMIT
    if truncated:
        text = text[: MEMORY_POST_TEXT_LIMIT - 1] + "…"
    footer = f"Capture {idea['id'][:8]} · Memory {memory_id}"
    if truncated:
        footer += " · Truncated"
    return slack_escape(f"[{priority.upper()}] Memory logged to {category}\n\n{text}\n\n{footer}")


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
    # Read the destination now: a config change during the Brain call must not turn
    # a saved fact into a refused promotion. Delivery re-checks the channel anyway.
    post_channel_id = slack_memory_posts.channel_id() if body.post_to_slack else ""
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


@router.get("/api/memory-inbox/review-health")
def review_health(days: int = 14, project: str = ""):
    """How local proposals were resolved in a bounded UTC window.

    Read-only and aggregate. It exists so the console can show whether a loop
    produces work worth doing without anyone reading every proposal, and every
    number it returns is reachable as a filter on the inbox itself.
    """
    if days < 1 or days > MAX_REVIEW_WINDOW_DAYS or len(project) > 80:
        raise HTTPException(status_code=422, detail="invalid_review_window")
    try:
        return inbox().review_health(days=days, project=project)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=503, detail="memory_inbox_unavailable") from exc


@router.post("/api/memory-inbox/{idea_id}/triage")
def triage_idea(idea_id: str, body: TriageRequest):
    if body.outcome not in REVIEW_OUTCOMES[1:]:
        raise HTTPException(status_code=422, detail="invalid_review_outcome")
    try:
        return {"idea": inbox().triage(idea_id, outcome=body.outcome, note=body.note)}
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
    if event.get("type") != "app_mention" or event.get("bot_id"):
        return {"ok": True}
    if event.get("subtype") not in (None, "", "file_share"):
        return {"ok": True}
    channel = event.get("channel")
    if not isinstance(channel, str) or channel not in channels:
        raise HTTPException(status_code=403, detail="slack_source_not_allowed")
    values = [payload.get("event_id"), event.get("user"), event.get("text")]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise HTTPException(status_code=400, detail="invalid_slack_payload")
    event_id, author, text = values
    files = event_files(event)
    thread_ts = ""
    if slack_receipts.enabled():
        thread_ts = event.get("thread_ts") or event.get("ts")
        if not isinstance(thread_ts, str) or not re.fullmatch(
            r"[0-9]{1,16}\.[0-9]{1,6}", thread_ts
        ):
            raise HTTPException(status_code=400, detail="invalid_slack_timestamp")
    # Match promotion's content check before creating an idea and its save receipt.
    # Acknowledge the event only; retries of mention-only messages have no side effects.
    # File-only mentions still save: the file header is the captured content.
    if not capture_text(text) and not files:
        return {"ok": True}
    contents = await fetch_file_contents(files)
    stored = build_captured_text(text, files, contents)
    key = "slack:event:" + hashlib.sha256(f"{team}:{event_id}".encode()).hexdigest()
    try:
        inbox().capture(
            text=stored,
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
    if not capture_text(fields["text"]):
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
