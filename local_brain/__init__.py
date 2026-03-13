"""
AIIA Local Brain — Mac Mini Intelligence Node — Home of AIIA

This package contains everything that runs on the local Mac Mini:
- AIIA (AI Interaction Architecture) — persistent AI teammate
- Ollama client for local LLM inference
- Smart Conductor for LLM-based query routing
- Background workers for summarization, memory extraction, PII scanning
- Local FastAPI service exposing these capabilities

Architecture:
    Mac Mini (Ollama + AIIA + Local Brain API)
        ↕ Tailscale tunnel
    Render (Production backend)

The production backend calls Local Brain for:
1. Smart routing (replaces keyword matching in Conductor)
2. Fast/cheap completions (summarization, memory extraction)
3. Embeddings (local nomic-embed-text)
4. PII scanning (compliance, runs locally for privacy)
5. AIIA queries (persistent knowledge + memory)
"""

from local_brain.config import LocalBrainConfig, get_config
from local_brain.ollama_client import OllamaClient

__all__ = ["LocalBrainConfig", "get_config", "OllamaClient"]
