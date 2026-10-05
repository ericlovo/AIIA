"""Create or update the Daily Digest Studio agent via the Command Center API.

Prints the payload by default. Pass ``--apply`` to POST/PATCH localhost:8200.
This never writes ``agent_data.json`` (or any other runtime store).

    python -m local_brain.scripts.ensure_daily_digest_agent
    python -m local_brain.scripts.ensure_daily_digest_agent --apply
    python -m local_brain.scripts.ensure_daily_digest_agent --apply --channel slack
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from local_brain.command_center.agent_output import DEFAULT_OUTPUT_CHANNEL, OUTPUT_CHANNELS
from local_brain.command_center.daily_digest import (
    DIGEST_AGENT_NAME,
    digest_agent_payload,
    find_digest_agent,
)

DEFAULT_CC = "http://127.0.0.1:8200"


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=8) as response:  # noqa: S310 - loopback Command Center only
        payload = json.loads(response.read().decode())
    return payload if isinstance(payload, dict) else {"value": payload}


def apply(base: str, payload: dict) -> dict:
    listed = _request("GET", f"{base.rstrip('/')}/api/agents")
    existing = find_digest_agent(listed.get("agents") or [])
    if existing:
        patched = _request("PATCH", f"{base.rstrip('/')}/api/agents/{existing['id']}", payload)
        return {"action": "updated", "agent": patched.get("agent") or patched}
    created = _request("POST", f"{base.rstrip('/')}/api/agents", payload)
    return {"action": "created", "agent": created.get("agent") or created}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="POST/PATCH the Command Center API. Default prints the payload only.",
    )
    parser.add_argument(
        "--channel",
        choices=OUTPUT_CHANNELS,
        default=DEFAULT_OUTPUT_CHANNEL,
        help="Declared output channel (Slack still falls back to the inbox).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_CC,
        help=f"Command Center base URL (default {DEFAULT_CC}).",
    )
    args = parser.parse_args(argv)
    payload = digest_agent_payload(output_channel=args.channel)
    if not args.apply:
        print(json.dumps({"agent": payload, "name": DIGEST_AGENT_NAME}, indent=2))
        print(
            "No files were written. Re-run with --apply on the Mini after pull/restart.",
            file=sys.stderr,
        )
        return 0
    try:
        result = apply(args.base_url, payload)
    except HTTPError as exc:
        print(f"Command Center rejected the request: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Command Center unreachable at {args.base_url}: {exc.reason}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
