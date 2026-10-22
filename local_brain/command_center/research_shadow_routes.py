"""Opt-in shadow cockpit; never mounted by the production server.

The host must supply an authenticated, server-bound tenant dependency. A URL,
header or body tenant identifier is not an authentication boundary.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from .research_shadow_store import ResearchShadowStore


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paused: bool


class KillControl(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    killed: bool


def shadow_router(store: ResearchShadowStore, tenant_dependency: Callable) -> APIRouter:
    """Host authenticates an operator and binds tenant; no default dependency."""
    router = APIRouter(prefix="/api/research-shadow", tags=["research-shadow"])

    def scope(tenant: Annotated[str, Depends(tenant_dependency)]) -> str:
        if not isinstance(tenant, str) or not tenant.startswith("synthetic-"):
            raise HTTPException(403, "synthetic_scope_required")
        return tenant

    def public(job: dict) -> dict:
        # Do not expose worker lease tokens, source text, raw provider responses,
        # secret references, checkpoints or credentials to the cockpit.
        config = job.get("config", {})
        return {
            "job_id": job["id"],
            "assignment_id": job["assignment_id"],
            "state": job["state"],
            "reason": job.get("reason")
            if job.get("reason")
            in {
                "cancelled",
                "budget_exhausted",
                "lease_expired",
                "provider_unavailable",
                "replay_validation_or_budget_failure",
                "replay_budget_exhausted",
                "recorded_provider_outage",
                "invalid_replay",
            }
            else None,
            "controls": store.get_controls(job["tenant_id"], job["assignment_id"]),
            "receipt": {
                key: job["receipt"][key]
                for key in (
                    "receipt_id",
                    "mode",
                    "review_status",
                    "approved",
                    "disposition",
                    "policy_version",
                    "usage",
                )
                if key in job["receipt"]
            }
            if job.get("receipt")
            else None,
            "attempts": job.get("attempts", 0),
            "mode": "replay-only",
            "configuration": {
                key: config[key]
                for key in (
                    "config_version",
                    "source_allowlist",
                    "geography",
                    "signal_taxonomy",
                    "schedule",
                    "enabled",
                    "model_ref",
                    "policy_version",
                    "budgets",
                    "retention_days",
                    "owner",
                )
                if key in config
            },
            "review": "human_review_required",
            "live_submission": False,
        }

    @router.get("/jobs")
    def jobs(tenant: Annotated[str, Depends(scope)]):
        return {"jobs": [public(job) for job in store.list_jobs(tenant)]}

    @router.get("/jobs/{job_id}")
    def job(job_id: str, tenant: Annotated[str, Depends(scope)]):
        result = store.get(tenant, job_id)
        if result is None:
            raise HTTPException(404, "job_not_found")
        return public(result)

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str, tenant: Annotated[str, Depends(scope)]):
        result = store.cancel(tenant, job_id)
        if result is None:
            raise HTTPException(404, "job_not_found")
        return public(result)

    @router.post("/assignments/{assignment_id}/pause")
    def pause(assignment_id: str, body: Control, tenant: Annotated[str, Depends(scope)]):
        store.set_paused(tenant, assignment_id, body.paused)
        return {"assignment_id": assignment_id, "paused": body.paused}

    @router.post("/kill-switch")
    def kill(body: KillControl, tenant: Annotated[str, Depends(scope)]):
        store.set_killed(tenant, body.killed)
        return {"killed": body.killed}

    return router
