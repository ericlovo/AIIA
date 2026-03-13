"""
Tests for the streaming chat fix — verifies that the CC dashboard chat
routes through /api/chat/stream (SSE) and that the Brain API's async
ChromaDB and reduced context window changes work correctly.

Run: pytest local_brain/tests/test_streaming_chat.py -v
Requires: Brain API on :8100, Command Center on :8200, Ollama running
"""

import asyncio
import json
import time

import httpx
import pytest

BRAIN_URL = "http://localhost:8100"
CC_URL = "http://localhost:8200"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def brain_healthy() -> bool:
    try:
        r = httpx.get(f"{BRAIN_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def cc_healthy() -> bool:
    try:
        r = httpx.get(f"{CC_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


requires_brain = pytest.mark.skipif(
    not brain_healthy(), reason="Brain API not running on :8100"
)
requires_cc = pytest.mark.skipif(
    not cc_healthy(), reason="Command Center not running on :8200"
)


# ──────────────────────────────────────────────
# Brain API tests
# ──────────────────────────────────────────────


@requires_brain
class TestBrainAPI:
    """Tests against the Brain API (:8100) directly."""

    def test_health_returns_async_stats(self):
        """Health endpoint should return knowledge_docs from async stats()."""
        r = httpx.get(f"{BRAIN_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "online"
        aiia = data.get("aiia", {})
        # knowledge_docs should be a positive int (async stats working)
        assert isinstance(aiia.get("knowledge", {}).get("knowledge_docs"), int)
        assert aiia["knowledge"]["knowledge_docs"] > 0

    def test_status_returns_async_stats(self):
        """Status endpoint should return knowledge + memory counts."""
        r = httpx.get(f"{BRAIN_URL}/v1/aiia/status", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data["knowledge"]["knowledge_docs"] > 0
        assert data["memory"]["total_memories"] > 0

    def test_ask_stream_returns_sse_events(self):
        """Streaming ask should return meta, chunk, and done SSE events."""
        events = {"meta": 0, "chunk": 0, "done": 0}
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                f"{BRAIN_URL}/v1/aiia/ask/stream",
                json={
                    "question": "hello",
                    "context": "",
                    "n_results": 1,
                    "max_tokens": 128,
                },
            ) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    evt = json.loads(line[6:])
                    evt_type = evt.get("type")
                    if evt_type in events:
                        events[evt_type] += 1

        assert events["meta"] >= 1, "Should get at least one meta event"
        assert events["chunk"] >= 1, "Should get at least one chunk"
        assert events["done"] == 1, "Should get exactly one done event"

    def test_ask_stream_num_ctx_respected(self):
        """Streaming ask with small num_ctx should still work (8192 default)."""
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                f"{BRAIN_URL}/v1/aiia/ask/stream",
                json={
                    "question": "hi",
                    "context": "",
                    "n_results": 1,
                    "max_tokens": 64,
                    "num_ctx": 4096,
                },
            ) as resp:
                assert resp.status_code == 200
                chunks = []
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        evt = json.loads(line[6:])
                        if evt.get("type") == "chunk":
                            chunks.append(evt["content"])
                assert len(chunks) > 0, "Should get chunks even with small ctx"

    def test_search_is_non_blocking(self):
        """Search endpoint (uses asyncio.to_thread for ChromaDB) should respond fast."""
        start = time.monotonic()
        r = httpx.post(
            f"{BRAIN_URL}/v1/aiia/search",
            json={"question": "test", "n_results": 1},
            timeout=10,
        )
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 5, f"Search took {elapsed:.1f}s — should be under 5s"

    def test_concurrent_status_and_search(self):
        """Status and search should work concurrently without blocking each other."""

        async def _run():
            async with httpx.AsyncClient(timeout=10) as client:
                results = await asyncio.gather(
                    client.get(f"{BRAIN_URL}/v1/aiia/status"),
                    client.post(
                        f"{BRAIN_URL}/v1/aiia/search",
                        json={"question": "test", "n_results": 1},
                    ),
                    client.get(f"{BRAIN_URL}/health"),
                )
                for r in results:
                    assert r.status_code == 200

        asyncio.run(_run())


# ──────────────────────────────────────────────
# Command Center tests
# ──────────────────────────────────────────────


@requires_cc
@requires_brain
class TestCommandCenterChat:
    """Tests against the Command Center (:8200) chat endpoints."""

    def test_stream_endpoint_returns_sse(self):
        """CC /api/chat/stream should proxy to Brain and return SSE events."""
        events = []
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                f"{CC_URL}/api/chat/stream",
                json={"message": "hello", "mode": "text"},
            ) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        assert "meta" in types, "Should have meta event"
        assert "chunk" in types, "Should have chunk events"
        assert "done" in types or any(
            e.get("type") == "chunk" for e in events
        ), "Should have content"

    def test_stream_saves_chat_history(self):
        """After a stream chat, the message should appear in history."""
        unique_msg = f"test-{int(time.time())}"

        # Send via stream
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                f"{CC_URL}/api/chat/stream",
                json={"message": unique_msg, "mode": "text"},
            ) as resp:
                # Consume the stream
                for _ in resp.iter_lines():
                    pass

        # Check history
        r = httpx.get(f"{CC_URL}/api/chat/history", timeout=5)
        assert r.status_code == 200
        history = r.json().get("history", [])
        user_msgs = [h["content"] for h in history if h["role"] == "user"]
        assert unique_msg in user_msgs, "Sent message should be in chat history"

    def test_dashboard_serves_streaming_js(self):
        """Dashboard HTML should reference /api/chat/stream, not just /api/chat."""
        r = httpx.get(f"{CC_URL}/", timeout=5)
        assert r.status_code == 200
        html = r.text
        assert "/api/chat/stream" in html, "Dashboard JS should use streaming endpoint"
        assert "stream-content" in html, "Dashboard should have stream content container"
