"""
AIIA Journal — mobile-friendly voice journaling pipeline.

Owns the audio → transcript → distillation → vault path used by:
- The aiia-console desktop app (calls keystore_transcribe + writes via vault_write)
- The /v1/journal/ingest HTTP endpoint (mobile clients, PWAs, etc.)
- The iCloud watcher (mobile-via-Obsidian-iCloud-sync, no VPN required)

Three integration points for mobile:

  1. iCloud watcher (recommended): drop a voice memo into
     ~/Library/Mobile Documents/com~apple~CloudDocs/AIIA-Inbox/ from your
     phone via an iOS Shortcut. The mini's watcher picks it up within ~5s,
     transcribes, distills, writes the markdown to the vault, then archives
     the source audio. No VPN needed — iCloud handles sync both directions.

  2. HTTP endpoint: POST /v1/journal/ingest with multipart audio. Direct,
     immediate, but requires the mini reachable from your phone (Tailscale
     or local Wi-Fi).

  3. Tauri-Mobile (future): the existing aiia-console codebase compiled for
     iOS would hit the same HTTP endpoint, or use the local keystore +
     vault paths directly when the Brain is running on the same device.

See JOURNAL.md for setup instructions.
"""

from local_brain.journal.pipeline import (
    JournalSession,
    JournalSessionResult,
    process_session,
)

__all__ = ["JournalSession", "JournalSessionResult", "process_session"]
