"""
Egress governance for AIIA.

Every cloud-bound call site asks authorize_egress() before dialing out.
Under AIIA_AIRGAP the decision is made locally — deny, except for the
explicit AIRGAP_ALLOWED_EGRESS allowlist (Voice Conductor / xai.realtime
ephemeral token mint) — and denied attempts are still reported to
Sanction so the denial lands in the audit trail. Outside air-gap the
decision comes from Sanction's /authorize/tool endpoint and fails
closed when Sanction is configured: timeout, transport error, or any
non-allow response ⇒ deny. When Sanction is not configured (vanilla
OSS install), egress is allowed — governance must never break core
AIIA functionality.

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
    "xai.realtime": "Voice Conductor ephemeral token",
}

PERMITTED_EGRESS = ["sanction control plane (metadata only)"]

# Intentional air-gap exceptions. Keep this set to Voice Conductor only —
# Studio/PWA holds the mic; the Mini mints an xAI ephemeral token. Do not
# add other cloud tools here; that would reopen the air-gap.
AIRGAP_ALLOWED_EGRESS = frozenset({"xai.realtime"})

_TIMEOUT = 5.0


def airgap_allows_tool(tool: str) -> bool:
    """True if this tool is on the air-gap exception allowlist."""
    return tool in AIRGAP_ALLOWED_EGRESS


@dataclass
class EgressDecision:
    allowed: bool
    reason: str


def _egress_state(name: str, enabled: bool) -> str:
    if not enabled:
        return "sanction-governed"
    if name in AIRGAP_ALLOWED_EGRESS:
        return "airgap-allowlisted"
    return "disabled"


def airgap_status(config=None) -> dict:
    """Airgap block for /health — enabled flag + per-egress-point status."""
    cfg = config or get_config()
    enabled = bool(getattr(cfg, "airgap_enabled", False))
    permitted = list(PERMITTED_EGRESS)
    if enabled:
        for name in sorted(AIRGAP_ALLOWED_EGRESS):
            label = EGRESS_POINTS.get(name, name)
            permitted.append(f"{name} ({label}; airgap exception)")
    return {
        "enabled": enabled,
        "egress": {name: _egress_state(name, enabled) for name in EGRESS_POINTS},
        "permitted": permitted,
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


def _airgap_decision(tool: str, server: str | None) -> EgressDecision | None:
    """Local air-gap decision, or None when air-gap is off (continue)."""
    if not get_config().airgap_enabled:
        return None
    if tool in AIRGAP_ALLOWED_EGRESS:
        return EgressDecision(True, f"allowed: air-gap exception ({tool})")
    report_denied_bg(tool, server)
    return EgressDecision(False, "denied: air-gap mode (AIIA_AIRGAP)")


async def authorize_egress(tool: str, server: str | None = None) -> EgressDecision:
    """Authorize a cloud egress attempt. Fail-closed when governed."""
    airgap = _airgap_decision(tool, server)
    if airgap is not None:
        return airgap
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
    airgap = _airgap_decision(tool, server)
    if airgap is not None:
        return airgap
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
