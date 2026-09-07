"""Command Center HTTP surface for Voice Conductor.

GET  /api/voice/status          — connected | not_configured (never returns the key)
POST /api/voice/session         — mint a short-lived xAI ephemeral token + session config
POST /api/voice/tools           — execute one allowlisted tool; fail-closed otherwise
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from local_brain.command_center.voice_conductor import (
    EGRESS_TOOL,
    EPHEMERAL_TTL_SECONDS,
    XAI_CLIENT_SECRETS_URL,
    XAI_REALTIME_URL,
    VoiceConductorDeps,
    execute_tool,
    load_xai_api_key,
    session_config,
    status_payload,
)
from local_brain.egress import authorize_egress

logger = logging.getLogger("aiia.voice_conductor.routes")


class VoiceToolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)


def build_voice_router(deps: VoiceConductorDeps) -> APIRouter:
    router = APIRouter(prefix="/api/voice", tags=["voice-conductor"])

    @router.get("/status")
    async def voice_status():
        return status_payload()

    @router.post("/session")
    async def voice_session():
        snapshot = status_payload()
        if snapshot["status"] != "connected":
            raise HTTPException(status_code=503, detail=snapshot["reason"] or "not_configured")

        decision = await authorize_egress(EGRESS_TOOL)
        if not decision.allowed:
            raise HTTPException(status_code=403, detail=f"EGRESS_DENIED:{decision.reason}")

        api_key = load_xai_api_key()
        if not api_key:
            raise HTTPException(status_code=503, detail="missing_xai_api_key")

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
                response = await client.post(
                    XAI_CLIENT_SECRETS_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"expires_after": {"seconds": EPHEMERAL_TTL_SECONDS}},
                )
        except httpx.HTTPError as exc:
            logger.warning("xAI ephemeral token request failed: %s", exc)
            raise HTTPException(status_code=502, detail="xai_token_unavailable") from exc

        if response.status_code >= 400:
            logger.warning("xAI ephemeral token HTTP %s", response.status_code)
            raise HTTPException(status_code=502, detail="xai_token_rejected")

        try:
            body = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="xai_token_invalid") from exc

        token = ""
        if isinstance(body, dict):
            raw = body.get("value") or body.get("token") or body.get("client_secret")
            if isinstance(raw, dict):
                raw = raw.get("value")
            if isinstance(raw, str):
                token = raw.strip()
        if not token:
            raise HTTPException(status_code=502, detail="xai_token_missing")

        expires_at = body.get("expires_at") if isinstance(body, dict) else None
        return {
            "status": "connected",
            "token": token,
            "expires_at": expires_at,
            "ttl_seconds": EPHEMERAL_TTL_SECONDS,
            "realtime_url": XAI_REALTIME_URL,
            "session": session_config(),
            "specialty": snapshot["specialty"],
        }

    @router.post("/tools")
    async def voice_tools(body: VoiceToolRequest):
        result = await execute_tool(body.name, body.arguments, deps)
        if not result.ok:
            raise HTTPException(status_code=result.status, detail=result.error)
        return result.to_dict()

    return router
