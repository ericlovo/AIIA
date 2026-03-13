#!/usr/bin/env python3
"""
Briefing CLI — fetch or generate AIIA's morning briefing from Command Center.

Usage:
    python -m local_brain.scripts.briefing_cli          # latest
    python -m local_brain.scripts.briefing_cli --fresh  # trigger + wait + fetch

Shell alias (add to Mini's .zshrc):
    alias briefing='python3 -m local_brain.scripts.briefing_cli'
    alias briefing-fresh='python3 -m local_brain.scripts.briefing_cli --fresh'
"""

import json
import sys
import time
import urllib.request
import urllib.error

CC_URL = "http://localhost:8200"


def fetch_latest():
    """Get the latest briefing from Command Center."""
    try:
        req = urllib.request.Request(f"{CC_URL}/api/briefing/latest")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": f"Command Center unreachable: {e}"}
    except Exception as e:
        return {"error": str(e)}


def trigger_generate():
    """Trigger a fresh briefing generation."""
    try:
        req = urllib.request.Request(
            f"{CC_URL}/api/briefing/generate",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def main():
    fresh = "--fresh" in sys.argv or "-f" in sys.argv

    if fresh:
        print("Triggering fresh briefing...")
        result = trigger_generate()
        if "error" in result:
            print(f"  Error: {result['error']}")
            return 1

        # Poll until done (max ~90s)
        for i in range(30):
            time.sleep(3)
            data = fetch_latest()
            status = data.get("status", "")
            if status == "done" and data.get("briefing"):
                break
            sys.stdout.write(".")
            sys.stdout.flush()
        print()
    else:
        data = fetch_latest()

    if "error" in data:
        print(f"Error: {data['error']}")
        return 1

    briefing = data.get("briefing", "")
    last_run = data.get("last_run", "never")
    status = data.get("status", "unknown")
    duration = data.get("duration_ms", 0)

    if not briefing:
        print("No briefing available. Run with --fresh to generate one.")
        return 0

    print(f"\n  AIIA Morning Briefing")
    print(f"  Generated: {last_run} | Status: {status} | {duration:.0f}ms")
    print(f"  {'=' * 52}\n")
    print(f"  {briefing.replace(chr(10), chr(10) + '  ')}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
