"""
Tests for Gemma4Capabilities detection in local_brain.config.

Verifies the detect() logic across model name variants and the
AIIA_NATIVE_TOOLS_ENABLED env var opt-in for native function calling.

Run: pytest local_brain/tests/test_phase2_config.py -v
"""

from __future__ import annotations

import os

import pytest

from local_brain.config import Gemma4Capabilities, LocalBrainConfig

# ─── Gemma4Capabilities.detect() — model name matching ────────────


class TestGemma4CapabilitiesDetect:
    def test_llama_model_returns_no_gemma_features(self, monkeypatch):
        """llama3.1 is not Gemma 4 — all flags should be False."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("llama3.1:8b-instruct-q8_0")
        assert caps.native_function_calling is False
        assert caps.native_audio_input is False
        assert caps.thinking_mode is False
        assert caps.native_system_prompt is False

    def test_e4b_without_opt_in_has_no_native_tools(self, monkeypatch):
        """E4B alone does NOT enable native tools — opt-in required."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("gemma4:e4b")
        assert caps.native_function_calling is False

    def test_e4b_with_opt_in_enables_native_tools(self, monkeypatch):
        """Setting AIIA_NATIVE_TOOLS_ENABLED=true flips the flag on."""
        monkeypatch.setenv("AIIA_NATIVE_TOOLS_ENABLED", "true")
        caps = Gemma4Capabilities.detect("gemma4:e4b")
        assert caps.native_function_calling is True

    def test_e4b_enables_native_audio(self, monkeypatch):
        """E4B has native audio input regardless of opt-in flag."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("gemma4:e4b")
        assert caps.native_audio_input is True

    def test_gemma4_26b_has_thinking_mode_but_no_audio(self, monkeypatch):
        """26B is Gemma 4 (thinking mode) but not E4B (no audio)."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("gemma4:26b")
        assert caps.thinking_mode is True
        assert caps.native_system_prompt is True
        assert caps.native_audio_input is False
        assert caps.native_function_calling is False

    def test_gemma4_with_dash_prefix_also_detected(self, monkeypatch):
        """Accept both 'gemma4' and 'gemma-4' naming conventions."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("gemma-4:e4b")
        assert caps.thinking_mode is True
        assert caps.native_audio_input is True

    def test_opt_in_on_non_e4b_has_no_effect(self, monkeypatch):
        """Opt-in flag only affects E4B — 26B doesn't have native tools."""
        monkeypatch.setenv("AIIA_NATIVE_TOOLS_ENABLED", "true")
        caps = Gemma4Capabilities.detect("gemma4:26b")
        assert caps.native_function_calling is False

    def test_opt_in_false_value_treated_as_off(self, monkeypatch):
        """AIIA_NATIVE_TOOLS_ENABLED=false keeps native tools off."""
        monkeypatch.setenv("AIIA_NATIVE_TOOLS_ENABLED", "false")
        caps = Gemma4Capabilities.detect("gemma4:e4b")
        assert caps.native_function_calling is False

    def test_opt_in_case_insensitive(self, monkeypatch):
        """AIIA_NATIVE_TOOLS_ENABLED=TRUE also works."""
        monkeypatch.setenv("AIIA_NATIVE_TOOLS_ENABLED", "TRUE")
        caps = Gemma4Capabilities.detect("gemma4:e4b")
        assert caps.native_function_calling is True

    def test_empty_model_name_is_safe(self, monkeypatch):
        """Empty model name should return the all-False default."""
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)
        caps = Gemma4Capabilities.detect("")
        assert caps.native_function_calling is False
        assert caps.native_audio_input is False
        assert caps.thinking_mode is False
        assert caps.native_system_prompt is False


# ─── LocalBrainConfig wiring ──────────────────────────────────────


class TestLocalBrainConfigCapabilitiesWiring:
    def test_config_populates_primary_capabilities_from_routing_model(self, monkeypatch):
        """LocalBrainConfig.primary_capabilities is populated from the
        routing model name at __post_init__ time."""
        monkeypatch.setenv("LOCAL_ROUTING_MODEL", "gemma4:e4b")
        monkeypatch.setenv("AIIA_NATIVE_TOOLS_ENABLED", "true")
        monkeypatch.delenv("AIIA_PRIMARY_MODEL", raising=False)

        config = LocalBrainConfig()

        assert config.primary_capabilities.native_function_calling is True
        assert config.primary_capabilities.native_audio_input is True
        assert config.primary_capabilities.thinking_mode is True

    def test_config_default_llama_has_no_gemma_features(self, monkeypatch):
        """Default config (llama3.1 routing) has all-False capabilities."""
        monkeypatch.delenv("LOCAL_ROUTING_MODEL", raising=False)
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)

        config = LocalBrainConfig()

        assert config.primary_capabilities.native_function_calling is False
        assert config.primary_capabilities.native_audio_input is False
        assert config.primary_capabilities.thinking_mode is False
        assert config.primary_capabilities.native_system_prompt is False

    def test_config_opt_out_is_default(self, monkeypatch):
        """Even with gemma4:e4b routing, native tools default to OFF."""
        monkeypatch.setenv("LOCAL_ROUTING_MODEL", "gemma4:e4b")
        monkeypatch.delenv("AIIA_NATIVE_TOOLS_ENABLED", raising=False)

        config = LocalBrainConfig()

        assert config.primary_capabilities.native_function_calling is False
        # But audio and thinking are still on (model-inherent, not opt-in)
        assert config.primary_capabilities.native_audio_input is True
        assert config.primary_capabilities.thinking_mode is True
