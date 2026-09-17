"""Fixed save and promotion receipts, delivered from a durable local outbox."""

import asyncio
import logging
import os
import sqlite3
import uuid

import httpx

from local_brain.egress import authorize_egress

logger = logging.getLogger(__name__)


def enabled():
    return os.getenv("AIIA_SLACK_ACK_ENABLED", "") == "1"


def configured():
    return enabled() and bool(os.getenv("AIIA_SLACK_BOT_TOKEN", ""))


async def deliver_one(inbox, *, transport=None):
    if not configured():
        return
    receipt = inbox.claim_receipt()
    if receipt is None:
        return
    team = os.getenv("AIIA_SLACK_TEAM_ID", "")
    channels = {c.strip() for c in os.getenv("AIIA_SLACK_CHANNEL_IDS", "").split(",")}
    if not team or receipt["workspace_id"] != team or receipt["channel_id"] not in channels:
        inbox.finish_receipt(receipt, status="failed", error="source_not_allowed")
        return
    decision = await authorize_egress("slack.capture_ack", server="slack.com")
    if not decision.allowed:
        inbox.finish_receipt(receipt, status="pending", error="egress_denied", delay=300)
        return
    # Never transmit the captured text or a caller-supplied URL/message body.
    kind = receipt.get("kind", "capture")
    if kind == "promotion":
        text = (
            "Logged to AIIA memory from the Mindmoor inbox. "
            f"Capture ID: {receipt['idea_id']}. Memory ID: {receipt.get('memory_id') or 'unrecorded'}"
        )
        message_key = receipt["idea_id"] + ":promotion"
    else:
        text = f"Saved to the local Mindmoor inbox for review. Capture ID: {receipt['idea_id']}"
        message_key = receipt["idea_id"]
    payload = {
        "channel": receipt["channel_id"],
        "thread_ts": receipt["thread_ts"],
        "text": text,
        "reply_broadcast": False,
        "unfurl_links": False,
        "unfurl_media": False,
        "mrkdwn": False,
        "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, message_key)),
    }
    inbox.finish_receipt(
        receipt, **await post_message(payload, receipt["attempts"], transport=transport)
    )


async def post_message(payload, attempts, *, transport=None):
    """One chat.postMessage attempt, classified for a durable outbox.

    Returns the finish fields: sent with the Slack ts, or pending/failed with a
    sanitized error code and backoff. Shared by receipts and memory posts so both
    keep the same 429 handling, permanent-error set and eight-attempt cap.
    """
    error, delay, permanent = (
        "delivery_unavailable",
        min(3600, 2 ** min(attempts, 10)),
        False,
    )
    try:
        async with httpx.AsyncClient(
            timeout=10, follow_redirects=False, transport=transport
        ) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={"Authorization": "Bearer " + os.environ["AIIA_SLACK_BOT_TOKEN"]},
            )
        if response.status_code == 429:
            error = "rate_limited"
            try:
                delay = max(delay, min(86400, int(response.headers.get("retry-after", "60"))))
            except ValueError:
                delay = max(delay, 60)
        elif response.status_code == 200:
            data = response.json()
            if (
                isinstance(data, dict)
                and data.get("ok") is True
                and isinstance(data.get("ts"), str)
            ):
                return {"status": "sent", "slack_ts": data["ts"]}
            code = data.get("error") if isinstance(data, dict) else None
            # Persist only known error codes, never token-bearing response bodies.
            if code in {
                "invalid_auth",
                "token_revoked",
                "missing_scope",
                "not_in_channel",
                "channel_not_found",
            }:
                error, permanent = code, True
        elif 400 <= response.status_code < 500:
            error, permanent = "request_rejected", True
    except (httpx.HTTPError, ValueError):
        pass
    return {
        "status": "failed" if permanent or attempts >= 8 else "pending",
        "error": error,
        "delay": delay,
    }


async def run_worker(inbox_factory):
    while True:
        try:
            if configured():
                await deliver_one(inbox_factory())
        except (OSError, sqlite3.Error):
            logger.warning("Slack receipt storage unavailable; retrying")
        except Exception:
            logger.warning("Slack receipt worker error; retrying")
        await asyncio.sleep(2)
