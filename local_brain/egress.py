"""
Egress governance for AIIA.

Every cloud-bound call site asks authorize_egress() before dialing out.
Under AIIA_AIRGAP the decision is made locally — always deny — and the
attempt is still reported to Sanction so the denial lands in the audit
trail. Outside air-gap the decision comes from Sanction's
/authorize/tool endpoint and fails closed when Sanction is configured:
timeout, transport error, or any non-allow response ⇒ deny. When
Sanction is not configured (vanilla OSS install), egress is allowed —
governance must never break core AIIA functionality.

A failure of the audit report itself never converts a deny into an
allow.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from local_brain.config import get_config
from local_brain.sanction import _client

logger = logging.getLogger("aiia.egress")

# Tool name → what it is for. Every cloud call site in the runtime is
# registered here; /health surfaces this list so an air-gapped install
# shows exactly which egress points are disabled.
EGRESS_POINTS = {
    "anthropic.messages": "journal distiller",
    "openai.messages": "journal distiller",
    "groq.messages": "journal distiller",
    "groq.whisper": "journal transcription",
    "slack.post": "Slack notify",
    "google.tts": "TTS synthesis",
    "anthropic.claude_code": "execution engine / story runner",
    "web.fetch": "research literature loop",
}

PERMITTED_EGRESS = ["sanction control plane (metadata only)"]

_TIMEOUT = 5.0


@dataclass
class EgressDecision:
    allowed: bool
    reason: str


def airgap_status(config=None) -> dict:
    """Airgap block for /health — enabled flag + per-egress-point status."""
    cfg = config or get_config()
    enabled = bool(getattr(cfg, "airgap_enabled", False))
    return {
        "enabled": enabled,
        "egress": {name: "disabled" if enabled else "sanction-governed" for name in EGRESS_POINTS},
        "permitted": PERMITTED_EGRESS,
    }


def _payload(tool: str, server: str | None, airgap: bool) -> dict:
    payload: dict = {"tool": tool}
    if server:
        payload["server"] = server
    if airgap:
        payload["arguments"] = {"airgap": True}
    return payload


def _decide(status_code: int, data: dict) -> EgressDecision:
    if status_code == 200 and data.get("authorized") is True:
        return EgressDecision(True, "allowed by sanction")
    reason = data.get("code") or data.get("reason") or f"HTTP {status_code}"
    return EgressDecision(False, f"denied: {reason}")


async def authorize_egress(tool: str, server: str | None = None) -> EgressDecision:
    """Authorize a cloud egress attempt. Fail-closed when governed."""
    if get_config().airgap_enabled:
        report_denied_bg(tool, server)
        return EgressDecision(False, "denied: air-gap mode (AIIA_AIRGAP)")
    sanction = _client()
    if sanction is None:
        return EgressDecision(True, "allowed: sanction not configured")
    api_url, api_key = sanction
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{api_url}/authorize/tool",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=_payload(tool, server, airgap=False),
            )
        data = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        return _decide(resp.status_code, data)
    except Exception as exc:
        logger.warning("egress %s: sanction unreachable, failing closed: %s", tool, exc)
        return EgressDecision(False, f"denied: sanction unreachable ({type(exc).__name__})")


def authorize_egress_sync(tool: str, server: str | None = None) -> EgressDecision:
    """Sync variant for non-async call sites (CLI entry points)."""
    if get_config().airgap_enabled:
        report_denied_bg(tool, server)
        return EgressDecision(False, "denied: air-gap mode (AIIA_AIRGAP)")
    sanction = _client()
    if sanction is None:
        return EgressDecision(True, "allowed: sanction not configured")
    api_url, api_key = sanction
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{api_url}/authorize/tool",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=_payload(tool, server, airgap=False),
            )
        data = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
        return _decide(resp.status_code, data)
    except Exception as exc:
        logger.warning("egress %s: sanction unreachable, failing closed: %s", tool, exc)
        return EgressDecision(False, f"denied: sanction unreachable ({type(exc).__name__})")


async def _report_denied(tool: str, server: str | None) -> None:
    """Post the denied attempt to Sanction for the audit trail. Fails silently."""
    sanction = _client()
    if sanction is None:
        return
    api_url, api_key = sanction
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(
                f"{api_url}/authorize/tool",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=_payload(tool, server, airgap=True),
            )
    except Exception as exc:
        logger.debug("egress audit report failed (suppressed): %s", exc)


def report_denied_bg(tool: str, server: str | None = None) -> None:
    """Schedule the audit report as a background task (sync-safe).

    With no running loop (CLI entry points) the report is posted
    synchronously, best-effort — still never blocking a deny on failure.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_report_denied(tool, server))
    except RuntimeError:
        sanction = _client()
        if sanction is None:
            return
        api_url, api_key = sanction
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                client.post(
                    f"{api_url}/authorize/tool",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json=_payload(tool, server, airgap=True),
                )
        except Exception as exc:
            logger.debug("egress audit report failed (suppressed): %s", exc)
