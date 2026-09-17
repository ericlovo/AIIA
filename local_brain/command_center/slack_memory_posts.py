"""Human-approved memory posts to one allowlisted Slack channel, from a durable outbox.

Unlike receipts, these carry captured text: only the body a human approved at
promote time, only to AIIA_SLACK_MEMORY_POST_CHANNEL_ID, and only when enabled.
"""

import asyncio
import logging
import os
import re
import sqlite3
import uuid

from local_brain.command_center.slack_receipts import post_message
from local_brain.egress import authorize_egress

logger = logging.getLogger(__name__)

CHANNEL_ID = re.compile(r"C[A-Z0-9]{8,}")


def enabled():
    return os.getenv("AIIA_SLACK_MEMORY_POST_ENABLED", "") == "1"


def channel_id():
    """The single outbound destination, or "" when unset or malformed."""
    value = os.getenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "").strip()
    return value if CHANNEL_ID.fullmatch(value) else ""


def configured():
    return (
        enabled()
        and bool(os.getenv("AIIA_SLACK_BOT_TOKEN", ""))
        and bool(os.getenv("AIIA_SLACK_TEAM_ID", ""))
        and bool(channel_id())
    )


async def deliver_one(inbox, *, transport=None):
    if not configured():
        return
    post = inbox.claim_memory_post()
    if post is None:
        return
    # The content goes back only to the workspace it came from.
    if post["workspace_id"] != os.environ["AIIA_SLACK_TEAM_ID"]:
        inbox.finish_memory_post(post, status="failed", error="source_not_allowed")
        return
    if post["channel_id"] != channel_id():
        inbox.finish_memory_post(post, status="failed", error="destination_not_allowed")
        return
    decision = await authorize_egress("slack.memory_post", server="slack.com")
    if not decision.allowed:
        inbox.finish_memory_post(post, status="pending", error="egress_denied", delay=300)
        return
    payload = {
        "channel": post["channel_id"],
        "text": post["body"],
        "reply_broadcast": False,
        "unfurl_links": False,
        "unfurl_media": False,
        "mrkdwn": False,
        "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, post["memory_id"] + ":memory_post")),
    }
    inbox.finish_memory_post(
        post, **await post_message(payload, post["attempts"], transport=transport)
    )


async def run_worker(inbox_factory):
    while True:
        try:
            if configured():
                await deliver_one(inbox_factory())
        except (OSError, sqlite3.Error):
            logger.warning("Slack memory post storage unavailable; retrying")
        except Exception:
            logger.warning("Slack memory post worker error; retrying")
        await asyncio.sleep(2)
