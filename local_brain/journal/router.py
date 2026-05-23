"""
FastAPI router for /v1/journal/* endpoints.

The HTTP path for mobile clients that can reach the mini directly (Tailscale,
local Wi-Fi, future PWA). For mobile clients that can't, see
local_brain.journal.watcher — the iCloud-folder path needs no networking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from local_brain.journal.distiller import DistillationError
from local_brain.journal.pipeline import JournalSession, process_session
from local_brain.journal.whisper import TranscriptionError

logger = logging.getLogger("aiia.journal.router")

router = APIRouter(prefix="/v1/journal", tags=["journal"])


@router.post("/ingest")
async def ingest_audio(
    audio: UploadFile = File(
        ..., description="Audio container — m4a, webm, mp3, wav, mp4 all accepted."
    ),
    duration_seconds: int = Form(
        0, description="Recording duration in seconds. Optional; frontmatter falls back to 0."
    ),
    started_at: str | None = Form(
        None,
        description="ISO-8601 timestamp of recording start. Optional; defaults to upload time.",
    ),
    language: str | None = Form(
        None, description="ISO-639-1 language hint (e.g. 'en') for Whisper. Optional."
    ),
):
    """Ingest an audio recording → transcribe → distill → write to vault.

    Returns the resulting vault path, the transcript, the distilled markdown,
    and whether distillation succeeded. On distillation failure, the raw
    transcript is still saved (frontmatter shape `tags: [journal, session, raw]`)
    and the response indicates `distilled=false`.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio body")

    if started_at:
        try:
            start = datetime.fromisoformat(started_at)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"started_at is not valid ISO-8601: {exc}",
            ) from exc
    else:
        start = datetime.now(tz=timezone.utc).astimezone()

    session = JournalSession(
        audio_bytes=audio_bytes,
        content_type=audio.content_type or "audio/mp4",
        started_at=start,
        duration_seconds=max(0, int(duration_seconds)),
        language=language,
        source_filename=audio.filename,
    )

    try:
        result = await process_session(session)
    except TranscriptionError as exc:
        # 422 — caller's audio was valid bytes but unusable upstream.
        raise HTTPException(status_code=422, detail=f"transcription failed: {exc}") from exc
    except DistillationError as exc:
        # This shouldn't fire — process_session catches it internally — but
        # guard anyway for forward-compat.
        raise HTTPException(status_code=502, detail=f"distillation failed: {exc}") from exc

    return {
        "vault_path": str(result.vault_path),
        "vault_relative": result.vault_relative,
        "distilled": result.distilled,
        "transcription": {
            "provider": "groq",
            "model": result.transcription_model,
            "text": result.transcript,
        },
        "distillation": (
            {
                "provider": result.distillation_provider,
                "model": result.distillation_model,
            }
            if result.distilled
            else None
        ),
        "markdown": result.markdown,
    }


@router.get("/health")
async def health():
    """Lightweight liveness check — used by mobile clients to confirm the
    Brain is reachable before attempting to upload audio."""
    import os

    from local_brain.journal.distiller import _pick_provider

    return {
        "status": "ok",
        "transcription_ready": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "distillation_ready": _pick_provider() is not None,
    }
