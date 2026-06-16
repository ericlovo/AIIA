"""
AutoFlux governance client for AIIA.

Fire-and-forget helpers that log LLM token usage to AutoFlux without
blocking the caller. All failures are suppressed — governance logging
must never break core AIIA functionality.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("aiia.autoflux")

_ANTHROPIC_COST_PER_M_IN = 3.00   # claude-sonnet-4-x input $/M tokens
_ANTHROPIC_COST_PER_M_OUT = 15.00  # claude-sonnet-4-x output $/M tokens


def _client() -> tuple[str, str] | None:
    """Return (api_url, api_key) or None if AutoFlux not configured."""
    url = os.getenv("SANCTION_API_URL", "").strip()
    key = os.getenv("SANCTION_API_KEY", "").strip()
    if not url or not key:
        return None
    return url, key


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    if "claude" in model:
        return (tokens_in * _ANTHROPIC_COST_PER_M_IN + tokens_out * _ANTHROPIC_COST_PER_M_OUT) / 1_000_000
    return 0.0


async def log_tokens(model: str, tokens_in: int, tokens_out: int, task: str | None = None) -> None:
    """Log LLM token usage to AutoFlux. Fails silently."""
    cfg = _client()
    if cfg is None:
        return
    api_url, api_key = cfg
    cost = _cost_usd(model, tokens_in, tokens_out)
    payload = {"model": model, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost}
    if task:
        payload["task"] = task
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{api_url}/tokens",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code == 402:
            logger.warning("AutoFlux: daily token budget exceeded")
        elif resp.status_code >= 400:
            logger.debug("AutoFlux token log failed: %s", resp.status_code)
    except Exception as exc:
        logger.debug("AutoFlux token log error (suppressed): %s", exc)


def log_tokens_bg(model: str, tokens_in: int, tokens_out: int, task: str | None = None) -> None:
    """Schedule token logging as a background task (sync-safe)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(log_tokens(model, tokens_in, tokens_out, task))
    except RuntimeError:
        pass
