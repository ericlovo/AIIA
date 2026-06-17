"""
Voice API — /v1/voice/* endpoints.

POST /v1/voice/transcribe  — upload audio blob, get back text
GET  /v1/voice/status      — whether whisper model is loaded
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from local_brain.voice.transcriber import transcribe_bytes, _model_size

logger = logging.getLogger("aiia.voice.router")

router = APIRouter(prefix="/v1/voice", tags=["voice"])


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    result = transcribe_bytes(data, filename=file.filename or "audio.webm")
    if result.get("error") and not result.get("text"):
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/status")
async def voice_status():
    from local_brain.voice.transcriber import _model
    return {
        "model": _model_size,
        "loaded": _model is not None,
    }
