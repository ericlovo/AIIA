"""
Groq Whisper transcription client.

Talks directly to https://api.groq.com/openai/v1/audio/transcriptions via
httpx (no SDK dep — the OpenAI-compatible REST shape is dead simple). Groq's
whisper-large-v3-turbo is the default; it's 10-20x real-time on a Mac mini's
network connection, free tier covers hours per day, and the quality is
indistinguishable from OpenAI's hosted Whisper in our journaling use case.

Mirrors the JS-side helper in aiia-console/src/journal/whisper.ts so a single
fix lands behavior in both places.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("aiia.journal.whisper")

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"


class TranscriptionError(RuntimeError):
    """Raised when the upstream Whisper call fails for any reason."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model: str
    provider: str = "groq"


def _resolve_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise TranscriptionError(
            "GROQ_API_KEY is not set. Add it to your environment (e.g. .env or "
            "your launchd plist) — the journal pipeline needs it for transcription."
        )
    return key


async def transcribe_audio(
    audio_bytes: bytes,
    *,
    content_type: str = "audio/mp4",
    filename: str | None = None,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    timeout_seconds: float = 90.0,
) -> TranscriptionResult:
    """Transcribe an audio blob via Groq Whisper.

    Args:
        audio_bytes: Raw container bytes (m4a, mp4, webm, mp3, wav, etc.).
        content_type: MIME type used for the multipart upload. Use whatever
            container the source file is in — Groq's parser handles all of
            them. iOS Shortcut recordings default to audio/mp4 (m4a).
        filename: Optional name to use in the multipart form. Cosmetic only.
        model: Whisper model id. whisper-large-v3-turbo is the fast default.
        language: Optional ISO-639-1 hint, e.g. "en". Improves accuracy on
            short / noisy recordings.
        timeout_seconds: httpx request timeout. Default 90s covers ~20-min
            sessions; bump for longer.

    Returns:
        TranscriptionResult with the transcribed text and the model used.

    Raises:
        TranscriptionError on any failure (missing key, HTTP error, empty
        response, malformed JSON).
    """
    api_key = _resolve_api_key()
    file_name = filename or "recording.m4a"

    form = {"model": (None, model), "response_format": (None, "json")}
    if language:
        form["language"] = (None, language)
    files = {"file": (file_name, audio_bytes, content_type)}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(
            GROQ_TRANSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={k: v[1] for k, v in form.items()},
            files=files,
        )

    if resp.status_code >= 400:
        # Cap the body so a giant HTML error page doesn't blow up logs.
        body = resp.text[:400]
        raise TranscriptionError(f"Groq Whisper returned HTTP {resp.status_code}: {body}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise TranscriptionError(f"Groq Whisper returned non-JSON body: {resp.text[:200]}") from exc

    text = (payload.get("text") or "").strip()
    if not text:
        raise TranscriptionError(
            "Groq Whisper returned an empty transcript — did the recording capture audio?"
        )

    logger.info(
        "transcribed audio: model=%s chars=%d audio_bytes=%d",
        model,
        len(text),
        len(audio_bytes),
    )
    return TranscriptionResult(text=text, model=model)
