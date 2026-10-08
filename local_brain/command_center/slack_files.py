"""Files attached to a capture mention, fetched into the Mini from a durable outbox.

A Slack event carries only file metadata. People attach a document and mention
the bot with no other text, so without this the capture is an empty mention.
The worker downloads each file with the bot token (inbound to the Mini; nothing
but the token leaves), saves it under the capture, extracts text when the type
allows, and appends an excerpt to the idea so it reads in the inbox and can be
promoted. Opt-in through AIIA_SLACK_FILE_CAPTURE_ENABLED, gated per request by
the `slack.file_fetch` egress point.
"""

import asyncio
import logging
import os
import re
import sqlite3
from pathlib import Path

import httpx

from local_brain.egress import authorize_egress

logger = logging.getLogger(__name__)

FILE_ID = re.compile(r"F[A-Z0-9]{8,}")
MAX_FILES_PER_CAPTURE = 10
MAX_FILE_BYTES = 5_000_000
EXCERPT_CHARS = 6_000
PDF_PAGE_LIMIT = 40
TEXT_PREFIXES = ("text/", "application/json", "application/x-yaml", "application/xml")
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".json", ".yml", ".yaml", ".log", ".toml"}
PERMANENT = {401: "invalid_auth", 403: "missing_scope", 404: "file_not_found"}
TRANSPORT = None  # tests inject an httpx transport; production dials Slack


class LookupDenied(Exception):
    pass


def enabled():
    return os.getenv("AIIA_SLACK_FILE_CAPTURE_ENABLED", "") == "1"


def configured():
    return enabled() and bool(os.getenv("AIIA_SLACK_BOT_TOKEN", ""))


def files_dir() -> Path:
    override = os.getenv("AIIA_CAPTURE_FILES_DIR", "")
    return Path(override) if override else Path(__file__).parent / "capture_files"


def _slack_url(url) -> bool:
    if not isinstance(url, str) or not url.startswith("https://"):
        return False
    try:
        host = httpx.URL(url).host
    except Exception:
        return False
    return host == "slack.com" or host.endswith(".slack.com")


def file_records(raw) -> list[dict]:
    """Validated metadata from an event's `files` list. Unknown shapes are dropped, never raised."""
    records = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        file_id = item.get("id")
        url = item.get("url_private_download") or item.get("url_private") or ""
        if not isinstance(file_id, str) or not FILE_ID.fullmatch(file_id) or not _slack_url(url):
            continue
        size = item.get("size")
        records.append(
            {
                "id": file_id,
                "name": str(item.get("name") or item.get("title") or file_id)[:200],
                "mimetype": str(item.get("mimetype") or "")[:100],
                "size": size if isinstance(size, int) and size >= 0 else 0,
                "url": url,
            }
        )
    return records[:MAX_FILES_PER_CAPTURE]


def _kb(size: int) -> str:
    return f"{max(1, round(size / 1024))} KB" if size else "size unknown"


def summary_lines(files: list[dict]) -> list[str]:
    """What the inbox shows before the bytes arrive."""
    return [
        f"[attached: {file['name']} ({file['mimetype'] or 'unknown type'}, {_kb(file['size'])})]"
        for file in files
    ]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:120] or "file"


def extract_text(path: Path, mimetype: str) -> str | None:
    """Plain text for the types we read; None means saved but not readable here."""
    suffix = path.suffix.lower()
    if mimetype == "application/pdf" or suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return None
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages[:PDF_PAGE_LIMIT])
        except Exception:
            return None
    if mimetype.startswith(TEXT_PREFIXES) or suffix in TEXT_SUFFIXES:
        return path.read_bytes().decode("utf-8", errors="replace")
    return None


def saved_text(record: dict) -> str:
    """Full extracted text of a fetched file, for indexing at promote time."""
    path = Path(record.get("path") or "")
    if record.get("status") != "done" or not path.is_file():
        return ""
    return (extract_text(path, record.get("mimetype") or "") or "").strip()


def _retry(error: str, attempts: int, delay: int) -> dict:
    return {"status": "failed" if attempts >= 8 else "pending", "error": error, "delay": delay}


async def fetch(file: dict, *, transport=None) -> dict:
    """One download attempt, classified for the outbox like post_message()."""
    attempts = file["attempts"]
    delay = min(3600, 2 ** min(attempts, 10))
    try:
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=True, transport=transport or TRANSPORT
        ) as client:
            response = await client.get(
                file["url"],
                headers={"Authorization": "Bearer " + os.environ["AIIA_SLACK_BOT_TOKEN"]},
            )
    except httpx.HTTPError:
        return _retry("delivery_unavailable", attempts, delay)
    if response.status_code in PERMANENT:
        return {"status": "failed", "error": PERMANENT[response.status_code]}
    if response.status_code == 429:
        return _retry("rate_limited", attempts, max(delay, 60))
    if response.status_code >= 500:
        return _retry("delivery_unavailable", attempts, delay)
    if response.status_code != 200:
        return {"status": "failed", "error": "request_rejected"}
    # Without files:read Slack answers 200 with its login page, not the file.
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html") and not file["mimetype"].startswith("text/html"):
        return {"status": "failed", "error": "missing_scope"}
    data = response.content
    if len(data) > MAX_FILE_BYTES:
        return {"status": "failed", "error": "file_too_large"}
    folder = files_dir() / file["idea_id"]
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    path = folder / f"{file['file_id']}-{safe_name(file['name'])}"
    path.write_bytes(data)
    path.chmod(0o600)
    text = extract_text(path, file["mimetype"])
    if text is None:
        note = f"(saved {_kb(len(data))}; no text extraction for {file['mimetype'] or 'this type'})"
        return {"status": "done", "path": str(path), "chars": 0, "excerpt": note}
    text = text.strip()
    if not text:
        return {"status": "done", "path": str(path), "chars": 0, "excerpt": "(file has no text)"}
    excerpt = text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1].rstrip() + "…"
    return {"status": "done", "path": str(path), "chars": len(text), "excerpt": excerpt}


async def deliver_one(inbox, *, transport=None):
    if not configured():
        return
    file = inbox.claim_file()
    if file is None:
        return
    team = os.getenv("AIIA_SLACK_TEAM_ID", "")
    if not team or file["workspace_id"] != team:
        inbox.finish_file(file, status="failed", error="source_not_allowed")
        return
    decision = await authorize_egress("slack.file_fetch", server="files.slack.com")
    if not decision.allowed:
        inbox.finish_file(file, status="pending", error="egress_denied", delay=300)
        return
    inbox.finish_file(file, **await fetch(file, transport=transport))


async def lookup(file_id: str, *, transport=None) -> dict | None:
    """files.info for a backfill: metadata only, same egress gate as the download."""
    decision = await authorize_egress("slack.file_fetch", server="slack.com")
    if not decision.allowed:
        raise LookupDenied()
    try:
        async with httpx.AsyncClient(timeout=10, transport=transport or TRANSPORT) as client:
            response = await client.get(
                "https://slack.com/api/files.info",
                params={"file": file_id},
                headers={"Authorization": "Bearer " + os.environ["AIIA_SLACK_BOT_TOKEN"]},
            )
        data = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("ok") is not True:
        return None
    records = file_records([data.get("file")])
    return records[0] if records else None


async def run_worker(inbox_factory):
    while True:
        try:
            if configured():
                await deliver_one(inbox_factory())
        except (OSError, sqlite3.Error):
            logger.warning("Slack file storage unavailable; retrying")
        except Exception:
            logger.warning("Slack file worker error; retrying")
        await asyncio.sleep(2)
