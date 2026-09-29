"""Public research jobs, mounted behind the same access boundary as Studio."""

import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from local_brain.command_center.public_signals import SignalJobs
from local_brain.command_center.slack_capture import inbox


def jobs():
    path = os.getenv("AIIA_SIGNAL_JOBS_PATH")
    return SignalJobs(Path(path) if path else inbox().path.with_name("signal_jobs.sqlite3"), inbox)


@asynccontextmanager
async def lifespan(app):
    worker = asyncio.create_task(jobs().loop())
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


router = APIRouter(lifespan=lifespan)


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


def error(exc):
    if isinstance(exc, ValueError):
        detail = str(exc)
        code = {"unknown_job": 404, "signals_busy": 409, "job_cooldown": 429}.get(detail, 503)
        return HTTPException(status_code=code, detail=detail)
    return HTTPException(status_code=503, detail="signal_storage_unavailable")


@router.get("/api/signal-jobs")
async def status():
    try:
        return jobs().status()
    except (sqlite3.Error, OSError) as exc:
        raise error(exc) from exc


@router.put("/api/signal-jobs/{job_id}")
async def configure(job_id: str, body: ScheduleRequest):
    try:
        service = jobs()
        service.configure(job_id, body.enabled)
        return service.status()
    except (ValueError, sqlite3.Error, OSError) as exc:
        raise error(exc) from exc


@router.post("/api/signal-jobs/{job_id}/run")
async def run(job_id: str):
    try:
        return await jobs().run(job_id)
    except (ValueError, sqlite3.Error, OSError) as exc:
        raise error(exc) from exc
