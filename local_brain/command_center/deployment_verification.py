"""Read explicit deployment verification receipts without probing or deploying."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{40}")


def verified_receipt(repo: str, now: datetime | None = None) -> dict | None:
    """Only the current production receipt is authoritative; never fall back."""
    filename = os.environ.get("AIIA_DEPLOYMENT_RECEIPTS")
    if not filename:
        return None
    try:
        path = Path(filename)
        if path.stat().st_size > 65536:
            return None
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            return None
        receipt = data["repositories"][repo]
        verified = datetime.fromisoformat(receipt["verified_at"].replace("Z", "+00:00"))
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        if verified.tzinfo is None or not timedelta(0) <= clock - verified <= timedelta(days=7):
            return None
        if (
            receipt["state"] != "verified"
            or receipt["environment"] not in {"production", "local-production"}
            or not _SHA.fullmatch(receipt["sha"])
            or receipt["health_passed"] is not True
            or receipt["browser_passed"] is not True
            or not isinstance(receipt["evidence"], str)
            or not receipt["evidence"].strip()
        ):
            return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
