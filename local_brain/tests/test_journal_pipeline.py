"""
Tests for local_brain.journal — distiller provider picking, pipeline filename
+ fallback shapes, watcher candidate detection.

We avoid hitting the real Groq / Anthropic APIs by patching the dispatch
table directly. The HTTP-layer integration is covered indirectly by the
existing test patterns in test_a2a_client.py (httpx.MockTransport) — when
the journal pipeline grows beyond Friday-night scope, mirror that approach.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_brain.journal import distiller, pipeline, watcher

# ──────────────────────────────────────────────────────────────────────────
# distiller — provider picker
# ──────────────────────────────────────────────────────────────────────────


def test_pick_provider_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    pick = distiller._pick_provider()
    assert pick is not None
    provider, model = pick
    assert provider == "anthropic"
    assert "claude" in model


def test_pick_provider_falls_back_to_openai_when_no_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    pick = distiller._pick_provider()
    assert pick is not None
    provider, _ = pick
    assert provider == "openai"


def test_pick_provider_falls_back_to_groq(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    pick = distiller._pick_provider()
    assert pick is not None
    provider, _ = pick
    assert provider == "groq"


def test_pick_provider_none_when_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert distiller._pick_provider() is None


def test_pick_provider_ignores_blank_env(monkeypatch):
    # Whitespace-only keys are equivalent to unset — common bug when a user
    # has `GROQ_API_KEY=` in a .env file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert distiller._pick_provider() is None


# ──────────────────────────────────────────────────────────────────────────
# distiller — prompt assembly
# ──────────────────────────────────────────────────────────────────────────


def test_system_prompt_includes_required_frontmatter_keys():
    started = datetime(2026, 5, 22, 21, 30, 0, tzinfo=timezone.utc)
    inp = distiller.DistillationInput(
        transcript="…",
        started_at=started,
        duration_seconds=423,
        transcription_provider="groq",
    )
    prompt = distiller._system_prompt(inp)
    assert "date: 2026-05-22" in prompt
    assert "started: 2026-05-22T21:30:00+00:00" in prompt
    assert "duration_seconds: 423" in prompt
    assert "transcription: groq" in prompt
    assert "aiia_managed: true" in prompt
    assert "tags: [journal, session]" in prompt


# ──────────────────────────────────────────────────────────────────────────
# pipeline — filename + fallback shape
# ──────────────────────────────────────────────────────────────────────────


def test_session_relative_path_format():
    started = datetime(2026, 5, 22, 21, 30, 7)
    rel = pipeline._session_relative_path(started)
    assert rel == "00-Inbox/2026-05-22-213007-session.md"


def test_fallback_markdown_contains_transcript_and_raw_tag():
    inp = distiller.DistillationInput(
        transcript="The thing on my mind is the move to the new place.",
        started_at=datetime(2026, 5, 22, 21, 30, 0, tzinfo=timezone.utc),
        duration_seconds=120,
        transcription_provider="groq",
    )
    md = pipeline._fallback_markdown(inp)
    assert md.startswith("---\n")
    assert "tags: [journal, session, raw]" in md
    assert "duration_seconds: 120" in md
    assert "The thing on my mind is the move to the new place." in md
    assert "*(distillation skipped — raw transcript below)*" in md


def test_fallback_markdown_has_aiia_managed_flag():
    inp = distiller.DistillationInput(
        transcript="hello",
        started_at=datetime(2026, 5, 22, 21, 30, 0),
        duration_seconds=0,
        transcription_provider="groq",
    )
    md = pipeline._fallback_markdown(inp)
    assert "aiia_managed: true" in md


# ──────────────────────────────────────────────────────────────────────────
# pipeline — atomic_write (uses real fs, isolated via tmp_path)
# ──────────────────────────────────────────────────────────────────────────


def test_atomic_write_creates_parents_and_writes(tmp_path):
    target = tmp_path / "00-Inbox" / "2026-05-22-session.md"
    pipeline._atomic_write(target, "# hello\n")
    assert target.exists()
    assert target.read_text() == "# hello\n"


def test_atomic_write_overwrites_existing(tmp_path):
    target = tmp_path / "note.md"
    pipeline._atomic_write(target, "first")
    pipeline._atomic_write(target, "second")
    assert target.read_text() == "second"


def test_atomic_write_leaves_no_tmp_artifact(tmp_path):
    target = tmp_path / "note.md"
    pipeline._atomic_write(target, "body")
    # tmp+rename means no `.md.tmp` remains after success.
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


# ──────────────────────────────────────────────────────────────────────────
# watcher — candidate detection
# ──────────────────────────────────────────────────────────────────────────


def test_watcher_picks_up_real_audio_file(tmp_path):
    f = tmp_path / "voice-memo.m4a"
    f.write_bytes(b"\x00" * 1024)
    assert watcher._is_audio_candidate(f) is True


def test_watcher_skips_icloud_placeholder(tmp_path):
    # iCloud's not-yet-downloaded placeholder pattern: leading dot, .icloud suffix.
    placeholder = tmp_path / ".voice-memo.m4a.icloud"
    placeholder.write_bytes(b"placeholder")
    assert watcher._is_audio_candidate(placeholder) is False


def test_watcher_skips_zero_byte(tmp_path):
    empty = tmp_path / "empty.m4a"
    empty.write_bytes(b"")
    assert watcher._is_audio_candidate(empty) is False


def test_watcher_skips_non_audio_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"hello")
    assert watcher._is_audio_candidate(f) is False


def test_watcher_skips_directory(tmp_path):
    sub = tmp_path / "archived"
    sub.mkdir()
    assert watcher._is_audio_candidate(sub) is False


def test_watcher_content_type_for_m4a(tmp_path):
    f = tmp_path / "test.m4a"
    f.write_bytes(b"x")
    ct = watcher._content_type_for(f)
    # mimetypes on macOS usually returns audio/mp4 for .m4a; we accept both.
    assert ct in ("audio/mp4", "audio/m4a")


def test_watcher_content_type_falls_back_to_mp4_for_unknown(tmp_path):
    f = tmp_path / "test.unknown"
    f.write_bytes(b"x")
    ct = watcher._content_type_for(f)
    assert ct == "audio/mp4"


# ──────────────────────────────────────────────────────────────────────────
# whisper — api key resolution
# ──────────────────────────────────────────────────────────────────────────


def test_whisper_resolve_key_raises_when_missing(monkeypatch):
    from local_brain.journal.whisper import TranscriptionError, _resolve_api_key

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(TranscriptionError) as exc:
        _resolve_api_key()
    assert "GROQ_API_KEY" in str(exc.value)


def test_whisper_resolve_key_trims_whitespace(monkeypatch):
    from local_brain.journal.whisper import _resolve_api_key

    monkeypatch.setenv("GROQ_API_KEY", "  gsk_test  ")
    assert _resolve_api_key() == "gsk_test"


def test_whisper_resolve_key_rejects_blank(monkeypatch):
    from local_brain.journal.whisper import TranscriptionError, _resolve_api_key

    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(TranscriptionError):
        _resolve_api_key()
