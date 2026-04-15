"""
Canonical Obsidian vault path resolution for AIIA.

Single source of truth for where AIIA reads and writes vault content on
disk. Every module that touches the vault should import from here
instead of reading OBSIDIAN_VAULT_DIR independently — migrating the
vault (e.g., into an Obsidian-synced iCloud Drive location) is then a
one-line env change instead of a multi-file edit.

Resolution order:
    1. OBSIDIAN_VAULT_DIR env var (preferred — set in .env or the host
       environment)
    2. ~/Documents/AIIA  (visible default: a standard Obsidian vault
       location that the AIIA Obsidian bridge can write to natively)
    3. ~/.aiia/vault  (hidden fallback: used when no user-visible
       vault exists, so AIIA still has somewhere to write)

The three-tier fallback exists so AIIA works out of the box on fresh
installs (no env var set, no Documents/AIIA created yet — lands in
~/.aiia/vault) while still letting operators point at a real Obsidian
vault via env var or by creating ~/Documents/AIIA.

Usage:

    from local_brain.vault_paths import vault_dir, vault_inbox

    VAULT_DIR = vault_dir()
    INBOX_DIR = vault_inbox()

    # Or derive subdirectories inline:
    decisions_path = vault_dir() / "30-Decisions" / "AIIA-Decisions.md"
"""

from __future__ import annotations

import os
from pathlib import Path

_VISIBLE_DEFAULT_REL = Path("Documents") / "AIIA"
_HIDDEN_FALLBACK_REL = Path(".aiia") / "vault"
_INBOX_SUBDIR = "00-Inbox"


def vault_dir() -> Path:
    """Return the canonical Obsidian vault path.

    Honors OBSIDIAN_VAULT_DIR if set. Otherwise prefers
    ~/Documents/AIIA (user-visible, Obsidian-friendly) if it already
    exists on disk. Otherwise falls back to ~/.aiia/vault (hidden,
    created lazily by consumers when they first write to it).

    The returned Path is fully expanded — no ~ or env vars remain.
    """
    env = os.getenv("OBSIDIAN_VAULT_DIR")
    if env:
        return Path(env).expanduser()

    home = Path.home()
    visible = home / _VISIBLE_DEFAULT_REL
    if visible.exists():
        return visible

    return home / _HIDDEN_FALLBACK_REL


def vault_inbox() -> Path:
    """Standard 00-Inbox subdirectory inside the vault.

    Consumers writing new daily notes, drafts, or captured ideas should
    land them here. The Obsidian bridge reads from here to import notes
    into AIIA's memory.
    """
    return vault_dir() / _INBOX_SUBDIR
