"""
iCloud watcher — the no-VPN mobile path for AIIA Journal.

On the mini, this runs as a launchd-supervised process. It polls a folder
inside iCloud Drive every few seconds; when a new audio file appears, it
runs the journaling pipeline and moves the source into an `archived/`
subfolder.

On the iPhone, the user creates a one-time iOS Shortcut that records audio
and saves it to that iCloud folder. The whole "mobile app" experience is
that Shortcut on the home screen — no Xcode, no App Store, no native code.

The polling interval is intentionally short (5s) so the round-trip from
"end of recording" to "markdown in vault" feels conversational. iCloud's
upload latency is the bottleneck, typically 10-60s on cellular.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from local_brain.journal.pipeline import JournalSession, process_session

logger = logging.getLogger("aiia.journal.watcher")


# Audio extensions the watcher will pick up. iOS Shortcut "Record Audio"
# default exports as m4a; "Voice Memo" defaults to m4a or mp3 depending on
# the user's settings.
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".wav", ".webm", ".ogg", ".flac"}

# Standard iCloud Drive root on macOS.
DEFAULT_ICLOUD_INBOX = (
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "AIIA-Inbox"
)


def _resolve_inbox() -> Path:
    """Honor AIIA_JOURNAL_WATCH_DIR override, else default iCloud path."""
    override = os.getenv("AIIA_JOURNAL_WATCH_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_ICLOUD_INBOX


def _archive_dir(inbox: Path) -> Path:
    return inbox / "archived"


def _content_type_for(path: Path) -> str:
    """Best-effort MIME lookup; falls back to audio/mp4 (the iOS default)."""
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("audio/"):
        return guessed
    if path.suffix.lower() == ".m4a":
        return "audio/mp4"
    return guessed or "audio/mp4"


def _is_audio_candidate(path: Path) -> bool:
    """Return True iff the path is a stable audio file ready to process."""
    if not path.is_file():
        return False
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        return False
    # Skip iCloud placeholder files (download not yet complete). These show
    # up as `.icloud` suffix and zero-byte placeholders during sync.
    name = path.name
    if name.startswith(".") and name.endswith(".icloud"):
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    # Stability check: if the file is still growing (download in progress),
    # defer until the next tick.
    try:
        first = path.stat().st_size
        time.sleep(0.5)
        second = path.stat().st_size
    except OSError:
        return False
    return first == second


def _started_at_from(path: Path) -> datetime:
    """Use the file's mtime as the session start. Phones write the file when
    the user finishes recording, so mtime ≈ session end. Subtract the
    duration when we know it; absent that, mtime is the best signal we have."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    except OSError:
        return datetime.now().astimezone()


async def _process_one(path: Path, archive: Path) -> None:
    """Run the pipeline for a single audio file. Move it to archive/ on
    success; leave it in place on failure so the next sweep retries."""
    logger.info("processing %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    audio_bytes = path.read_bytes()
    content_type = _content_type_for(path)
    # We don't actually know the recording's duration from the file alone
    # without parsing the container. Use a 0 placeholder; the markdown
    # frontmatter still gets `started:` from mtime.
    session = JournalSession(
        audio_bytes=audio_bytes,
        content_type=content_type,
        started_at=_started_at_from(path),
        duration_seconds=0,
        source_filename=path.name,
    )
    try:
        result = await process_session(session)
    except Exception:
        logger.exception("pipeline failed for %s", path.name)
        return

    # Move source into archive/ with the result-path tag for traceability.
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / path.name
    # If the same filename already exists (very unlikely with iOS timestamps),
    # suffix with the result's stem.
    if target.exists():
        target = archive / f"{path.stem}-{result.vault_path.stem}{path.suffix}"
    os.rename(str(path), str(target))
    logger.info(
        "completed %s → %s (distilled=%s)",
        path.name,
        result.vault_relative,
        result.distilled,
    )


async def _sweep(inbox: Path) -> int:
    """One pass over the inbox. Returns the number of files processed."""
    if not inbox.exists():
        return 0
    archive = _archive_dir(inbox)
    processed = 0
    for entry in sorted(inbox.iterdir(), key=lambda p: p.name):
        if entry == archive:
            continue
        if not _is_audio_candidate(entry):
            continue
        await _process_one(entry, archive)
        processed += 1
    return processed


async def watch_forever(
    *,
    poll_interval_seconds: float = 5.0,
    inbox: Path | None = None,
) -> None:
    """Long-running loop. Polls the inbox at the configured cadence and runs
    the pipeline on each new audio file. Designed to be supervised by
    launchd — exits silently on SIGTERM."""
    target = inbox or _resolve_inbox()
    logger.info("journal watcher starting: inbox=%s interval=%.1fs", target, poll_interval_seconds)
    target.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            await _sweep(target)
        except Exception:
            logger.exception("sweep failed (continuing)")
        await asyncio.sleep(poll_interval_seconds)


def main() -> None:
    """Entry point for `python -m local_brain.journal.watcher` and the
    `aiia journal-watch` CLI command."""
    logging.basicConfig(
        level=os.getenv("AIIA_JOURNAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(watch_forever())
    except KeyboardInterrupt:
        logger.info("journal watcher stopped")


if __name__ == "__main__":
    main()
