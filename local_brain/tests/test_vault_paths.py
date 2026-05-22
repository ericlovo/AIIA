"""
Tests for local_brain/vault_paths.py.

Pure env-var + filesystem resolution — uses tmp_path for the fallback
chain so no real home directory state is touched.

Run: pytest local_brain/tests/test_vault_paths.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_brain import vault_paths

# ─── vault_dir() — env var path ─────────────────────────────────────


def test_vault_dir_honors_obsidian_vault_dir(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "/tmp/test-vault")
    assert vault_paths.vault_dir() == Path("/tmp/test-vault")


def test_vault_dir_expands_tilde(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "~/CustomVault")
    result = vault_paths.vault_dir()
    assert result == Path.home() / "CustomVault"
    assert "~" not in str(result)


def test_vault_dir_env_var_wins_even_when_visible_default_exists(monkeypatch, tmp_path):
    """OBSIDIAN_VAULT_DIR takes precedence over the visible default."""
    # Point Path.home() at a tmp dir that DOES have Documents/AIIA
    monkeypatch.setattr(vault_paths.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / "Documents" / "AIIA").mkdir(parents=True)

    # Env var should still win
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "/tmp/explicit-override")
    assert vault_paths.vault_dir() == Path("/tmp/explicit-override")


# ─── vault_dir() — visible default (~/Documents/AIIA) ──────────────


def test_vault_dir_uses_visible_default_when_it_exists(monkeypatch, tmp_path):
    """If ~/Documents/AIIA exists and env var is unset, use it."""
    monkeypatch.delenv("OBSIDIAN_VAULT_DIR", raising=False)
    monkeypatch.setattr(vault_paths.Path, "home", classmethod(lambda cls: tmp_path))

    visible = tmp_path / "Documents" / "AIIA"
    visible.mkdir(parents=True)

    assert vault_paths.vault_dir() == visible


# ─── vault_dir() — hidden fallback (~/.aiia/vault) ──────────────────


def test_vault_dir_falls_back_to_hidden_when_visible_missing(monkeypatch, tmp_path):
    """If neither env var nor ~/Documents/AIIA exists, fall back."""
    monkeypatch.delenv("OBSIDIAN_VAULT_DIR", raising=False)
    monkeypatch.setattr(vault_paths.Path, "home", classmethod(lambda cls: tmp_path))
    # ~/Documents/AIIA does NOT exist in tmp_path

    result = vault_paths.vault_dir()
    assert result == tmp_path / ".aiia" / "vault"


def test_vault_dir_empty_env_treated_as_unset(monkeypatch, tmp_path):
    """Empty string OBSIDIAN_VAULT_DIR falls through to normal resolution."""
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "")
    monkeypatch.setattr(vault_paths.Path, "home", classmethod(lambda cls: tmp_path))

    result = vault_paths.vault_dir()
    # No visible default, no env — falls back to hidden
    assert result == tmp_path / ".aiia" / "vault"


# ─── vault_inbox() ──────────────────────────────────────────────────


def test_vault_inbox_appends_00_inbox(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "/tmp/test-vault")
    assert vault_paths.vault_inbox() == Path("/tmp/test-vault/00-Inbox")


def test_vault_inbox_tracks_vault_dir_changes(monkeypatch, tmp_path):
    """vault_inbox() re-derives on every call, so changes to the vault
    root propagate automatically."""
    monkeypatch.delenv("OBSIDIAN_VAULT_DIR", raising=False)
    monkeypatch.setattr(vault_paths.Path, "home", classmethod(lambda cls: tmp_path))

    # Initially: no visible default → hidden fallback
    assert vault_paths.vault_inbox() == tmp_path / ".aiia" / "vault" / "00-Inbox"

    # Create visible default → inbox resolves there now
    visible = tmp_path / "Documents" / "AIIA"
    visible.mkdir(parents=True)
    assert vault_paths.vault_inbox() == visible / "00-Inbox"


# ─── Integration: consumers get consistent resolution ─────────────


def test_vault_dir_is_a_pathlib_path(monkeypatch):
    """The helper always returns a Path, never a str."""
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "/tmp/test-vault")
    result = vault_paths.vault_dir()
    assert isinstance(result, Path)


def test_vault_inbox_is_a_pathlib_path(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", "/tmp/test-vault")
    result = vault_paths.vault_inbox()
    assert isinstance(result, Path)
