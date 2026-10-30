"""Operator-local conductor API. Tenant scope comes only from versioned config."""

import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .bd_conductor import BDConductor
from .bd_pipeline import DataUnavailable, PublicExcerpt, load_public_excerpt
from .bd_service import BDService


class SourceImport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    bot_id: str = Field(min_length=1, max_length=100)
    source: PublicExcerpt


class KillControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


class ReviewControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["human_review_required", "research_next", "watch", "dismissed"]


class ScheduleControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


class ShadowControl(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SteerControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "pause",
        "resume",
        "pause_bot",
        "resume_bot",
        "focus",
        "clear_focus",
        "bump",
        "queue",
        "set_cadence",
        "set_budget",
        "kill_switch",
        "cancel",
    ]
    bot_id: str | None = Field(default=None, min_length=1, max_length=100)
    job_id: str | None = Field(default=None, min_length=1, max_length=64)
    segment: str | None = Field(default=None, min_length=1, max_length=100)
    query: str | None = Field(default=None, min_length=1, max_length=200)
    target: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: StrictBool | None = None
    overnight_enabled: StrictBool | None = None
    around_the_clock: StrictBool | None = None
    weekday_interval_seconds: int | None = Field(default=None, ge=60, le=86400)
    weekend_interval_seconds: int | None = Field(default=None, ge=60, le=86400)
    overnight_interval_seconds: int | None = Field(default=None, ge=60, le=86400)
    tokens: int | None = Field(default=None, gt=0)
    calls: int | None = Field(default=None, gt=0)
    seconds: int | None = Field(default=None, gt=0)


def same_origin(request: Request):
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "same_origin_required")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "same_origin_required")


def create_router(factory=None):
    instance = None

    def service():
        nonlocal instance
        if instance is None:
            if factory:
                instance = factory()
            else:
                path = Path(
                    os.getenv(
                        "AIIA_BD_DATABASE",
                        str(Path(__file__).parent / "bd_data" / "conductor.sqlite3"),
                    )
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                instance = BDService(BDConductor(path, os.getenv("AIIA_BD_CONFIG")))
        return instance

    @asynccontextmanager
    async def lifespan(app):
        await ready()
        try:
            yield
        finally:
            if instance:
                await instance.stop()

    router = APIRouter(prefix="/api/bd", tags=["bd"], lifespan=lifespan)

    async def ready():
        try:
            result = service()
            await result.start()
            return result
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(503, "bd_storage_or_configuration_unavailable") from exc

    @router.get("/snapshot")
    async def snapshot(current: BDService = Depends(ready)):
        return current.snapshot()

    @router.post("/shadow", dependencies=[Depends(same_origin)])
    async def shadow(body: ShadowControl, current: BDService = Depends(ready)):
        try:
            return {"job_ids": current.enqueue_cycle(shadow=True), "mode": "replay"}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/sources", status_code=202, dependencies=[Depends(same_origin)])
    async def import_source(body: SourceImport, current: BDService = Depends(ready)):
        store = current.store
        if store.config["mode"] != "live":
            raise HTTPException(409, "source_import_requires_live_mode")
        bot = store.bots.get(body.bot_id)
        if not bot or not bot["enabled"]:
            raise HTTPException(404, "enabled_bot_not_found")
        if bot.get("kind") != "segment" or bot.get("segment") == "client-bd":
            raise HTTPException(409, "managed_leads_tenant_api_unavailable")
        if store.snapshot()["kill_switch"]:
            raise HTTPException(409, "conductor_stopped")
        payload = body.source.model_dump()
        try:
            load_public_excerpt(payload, bot.get("source_allowlist", []))
        except ValueError as exc:
            code = exc.code if isinstance(exc, DataUnavailable) else "invalid_public_source"
            raise HTTPException(422, code) from None
        entity = hashlib.sha256(
            (payload["url"] + "\0" + payload["source_hash"]).encode()
        ).hexdigest()
        queued = store.enqueue(
            bot["id"],
            bot["tenant_id"],
            entity,
            "public-source-research",
            payload={"public_source": payload},
            mode="live",
        )
        return {
            "job_id": queued["id"],
            "tenant_id": bot["tenant_id"],
            "mode": "live",
            "source_kind": "imported_primary_excerpt",
        }

    @router.post("/kill-switch", dependencies=[Depends(same_origin)])
    async def kill(body: KillControl, current: BDService = Depends(ready)):
        current.set_kill_switch(body.enabled)
        return current.snapshot()

    @router.post("/schedule", dependencies=[Depends(same_origin)])
    async def schedule(body: ScheduleControl, current: BDService = Depends(ready)):
        current.set_schedule_enabled(body.enabled)
        return current.snapshot()

    @router.post("/steer", dependencies=[Depends(same_origin)])
    async def steer(body: SteerControl, current: BDService = Depends(ready)):
        payload = body.model_dump()
        action = payload.pop("action")
        present = {key for key, value in payload.items() if value is not None}
        allowed = {
            "pause": set(),
            "resume": set(),
            "pause_bot": {"bot_id"},
            "resume_bot": {"bot_id"},
            "focus": {"bot_id", "segment"},
            "clear_focus": set(),
            "bump": {"job_id"},
            "queue": {"bot_id", "query", "target"},
            "set_cadence": {
                "bot_id",
                "overnight_enabled",
                "around_the_clock",
                "weekday_interval_seconds",
                "weekend_interval_seconds",
                "overnight_interval_seconds",
            },
            "set_budget": {"bot_id", "tokens", "calls", "seconds"},
            "kill_switch": {"enabled"},
            "cancel": {"job_id"},
        }
        extra = present - allowed.get(action, set())
        if extra:
            raise HTTPException(422, "unexpected_steer_field")
        required = {
            "pause_bot": ("bot_id",),
            "resume_bot": ("bot_id",),
            "bump": ("job_id",),
            "queue": ("bot_id",),
            "kill_switch": ("enabled",),
            "cancel": ("job_id",),
        }
        for field in required.get(action, ()):
            if payload.get(field) is None:
                raise HTTPException(422, f"{field}_required")
        if action == "focus" and (payload.get("bot_id") is None) == (
            payload.get("segment") is None
        ):
            raise HTTPException(422, "focus_requires_bot_or_segment")
        if action == "queue" and payload.get("query") is None and payload.get("target") is None:
            raise HTTPException(422, "queue_requires_query_or_target")
        if (
            action == "set_cadence"
            and payload.get("around_the_clock") is not None
            and payload.get("bot_id") is None
        ):
            raise HTTPException(422, "bot_id_required")
        if action == "kill_switch" and payload.get("enabled") is None:
            raise HTTPException(422, "enabled_required")
        try:
            return current.steer(action, **payload)
        except KeyError as exc:
            raise HTTPException(404, str(exc) or "not_found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/packets/{job_id}/{index}/review", dependencies=[Depends(same_origin)])
    async def review(
        job_id: str, index: int, body: ReviewControl, current: BDService = Depends(ready)
    ):
        try:
            current.store.set_packet_review(job_id, index, body.status)
        except KeyError as exc:
            raise HTTPException(404, "packet_not_found") from exc
        except ValueError as exc:
            raise HTTPException(409, "packet_not_reviewable") from exc
        return current.snapshot()

    @router.get("/events")
    async def events(request: Request, current: BDService = Depends(ready)):
        async def stream():
            previous = None
            while not await request.is_disconnected():
                snapshot = json.dumps(current.snapshot(), separators=(",", ":"))
                if snapshot != previous:
                    yield f"event: snapshot\ndata: {snapshot}\n\n"
                    previous = snapshot
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return router
