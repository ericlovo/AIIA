"""
End-to-end journal session pipeline.

Composes the three phases — transcribe, distill, write — into a single
function `process_session` that any client (HTTP endpoint, iCloud watcher,
or CLI command) can call with raw audio bytes.

The flow is identical to the desktop console's wrap pipeline:
  1. Transcribe via Groq Whisper.
  2. Write a raw-transcript fallback to the vault immediately, so the
     session is never lost to a flaky distillation call.
  3. Distill via the configured chat provider.
  4. Overwrite the file with the distilled markdown.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from local_brain.journal.distiller import (
    DistillationError,
    DistillationInput,
    distill,
)
from local_brain.journal.whisper import (
    TranscriptionError,
    transcribe_audio,
)
from local_brain.vault_paths import vault_dir

logger = logging.getLogger("aiia.journal.pipeline")


@dataclass(frozen=True)
class JournalSession:
    audio_bytes: bytes
    content_type: str
    started_at: datetime
    duration_seconds: int
    language: str | None = None
    source_filename: str | None = None  # cosmetic; used in multipart


@dataclass(frozen=True)
class JournalSessionResult:
    vault_path: Path
    vault_relative: str
    transcript: str
    markdown: str
    distilled: bool
    transcription_model: str
    distillation_provider: str | None
    distillation_model: str | None


def _session_relative_path(started_at: datetime) -> str:
    """00-Inbox/YYYY-MM-DD-HHMMSS-session.md — matches the console convention."""
    stamp = started_at.strftime("%Y-%m-%d-%H%M%S")
    return f"00-Inbox/{stamp}-session.md"


def _fallback_markdown(inp: DistillationInput) -> str:
    """Frontmatter-only emergency-save shape — guarantees no session is lost
    even when distillation fails. Matches the desktop console's fallback."""
    started_compact = inp.started_at.isoformat(sep=" ")[:16]
    return (
        "---\n"
        f"date: {inp.started_at.date().isoformat()}\n"
        f"started: {inp.started_at.isoformat()}\n"
        f"duration_seconds: {inp.duration_seconds}\n"
        f"transcription: {inp.transcription_provider}\n"
        "aiia_managed: true\n"
        "tags: [journal, session, raw]\n"
        "---\n"
        "\n"
        f"# Session {started_compact}\n"
        "\n"
        "*(distillation skipped — raw transcript below)*\n"
        "\n"
        f"{inp.transcript}\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    """Write atomically via tmp+rename — safe for iCloud and Obsidian-watched
    folders. Same primitive vault_writer uses for category cluster files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.rename(str(tmp), str(path))


async def process_session(session: JournalSession) -> JournalSessionResult:
    """Run the full journal pipeline for a session.

    Order of operations is chosen so a failure in the LLM step doesn't lose
    the session: the raw transcript is durably saved BEFORE distillation.
    """
    # Phase 1 — transcribe.
    try:
        transcription = await transcribe_audio(
            session.audio_bytes,
            content_type=session.content_type,
            filename=session.source_filename,
            language=session.language,
        )
    except TranscriptionError:
        raise
    transcript = transcription.text

    # Phase 2 — write the raw-transcript fallback. This is the "safety net"
    # write; the file may be overwritten by the distilled version below.
    distill_input = DistillationInput(
        transcript=transcript,
        started_at=session.started_at,
        duration_seconds=session.duration_seconds,
        transcription_provider=transcription.provider,
    )
    relative = _session_relative_path(session.started_at)
    absolute = vault_dir() / relative
    _atomic_write(absolute, _fallback_markdown(distill_input))
    logger.info("wrote raw-transcript fallback: %s", absolute)

    # Phase 3 — distill. Best-effort: on failure, keep the fallback in place
    # and surface the error to the caller as a partial-success result.
    try:
        distillation = await distill(distill_input)
    except DistillationError as exc:
        logger.warning("distillation skipped: %s", exc)
        return JournalSessionResult(
            vault_path=absolute,
            vault_relative=relative,
            transcript=transcript,
            markdown=_fallback_markdown(distill_input),
            distilled=False,
            transcription_model=transcription.model,
            distillation_provider=None,
            distillation_model=None,
        )

    # Phase 4 — overwrite the fallback with the distilled markdown.
    _atomic_write(absolute, distillation.markdown)
    logger.info("wrote distilled session: %s", absolute)

    return JournalSessionResult(
        vault_path=absolute,
        vault_relative=relative,
        transcript=transcript,
        markdown=distillation.markdown,
        distilled=True,
        transcription_model=transcription.model,
        distillation_provider=distillation.provider,
        distillation_model=distillation.model,
    )
