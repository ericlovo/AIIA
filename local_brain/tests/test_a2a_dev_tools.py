"""
Tests for the new A2A dev-tool agents:
- AIIAMemoryExecutor (remember + search modes)
- bootstrap registration of all 6 new agents
- tag filtering across the registered set

Mocks AIIA via a minimal stub so tests run without ChromaDB / Ollama init.

Run: pytest aiia/local_brain/tests/test_a2a_dev_tools.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from local_brain.a2a.bootstrap import _find_brain_cli, register_default_agents
from local_brain.a2a.executors.aiia_memory_executor import (
    AIIAMemoryExecutor,
    _parse_tagged_fact,
)
from local_brain.a2a.schema import Message, TextPart

# ─── Stubs ───────────────────────────────────────────────────────────


class _StubKnowledge:
    """Minimal knowledge store stand-in supporting search()."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[tuple] = []

    async def search(self, query: str, n_results: int = 5):
        self.calls.append((query, n_results))
        return self._results


class _StubAIIA:
    """Minimal AIIA stand-in exposing remember() and _knowledge."""

    def __init__(self, knowledge: _StubKnowledge | None = None) -> None:
        self._knowledge = knowledge
        self.remember_calls: list[dict[str, Any]] = []

    async def remember(
        self,
        fact: str,
        category: str = "lessons",
        source: str = "session",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "id": "mem-1",
            "fact": fact,
            "category": category,
            "source": source,
            "metadata": metadata or {},
        }
        self.remember_calls.append(entry)
        return entry


def _msg(text: str, role: str = "user") -> Message:
    return Message(role=role, parts=[TextPart(text=text)])


async def _stub_getter_factory(aiia):
    async def _getter():
        return aiia

    return _getter


# ─── _parse_tagged_fact ─────────────────────────────────────────────


def test_parse_tagged_fact_recognizes_valid_category():
    cat, fact = _parse_tagged_fact("decisions: pin chromadb to 0.5.23", default="lessons")
    assert cat == "decisions"
    assert fact == "pin chromadb to 0.5.23"


def test_parse_tagged_fact_unknown_prefix_falls_through():
    cat, fact = _parse_tagged_fact("https://example.com is the docs", default="lessons")
    assert cat == "lessons"
    assert fact == "https://example.com is the docs"


def test_parse_tagged_fact_no_colon_uses_default():
    cat, fact = _parse_tagged_fact("a fact with no prefix", default="patterns")
    assert cat == "patterns"
    assert fact == "a fact with no prefix"


def test_parse_tagged_fact_empty_after_colon_falls_through():
    cat, fact = _parse_tagged_fact("decisions:   ", default="lessons")
    assert cat == "lessons"
    assert fact == "decisions:"


# ─── AIIAMemoryExecutor — remember mode ─────────────────────────────


@pytest.mark.asyncio
async def test_remember_executor_writes_default_category():
    aiia = _StubAIIA()
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="remember")

    result = await executor.execute(_msg("we pinned httpx to 0.27.2"))

    assert len(aiia.remember_calls) == 1
    call = aiia.remember_calls[0]
    assert call["fact"] == "we pinned httpx to 0.27.2"
    assert call["category"] == "lessons"
    assert call["source"] == "a2a"

    assert result.artifact_name == "memory-entry"
    assert "AIIA remembered" in result.parts[0].text
    assert result.metadata["category"] == "lessons"
    assert result.metadata["id"] == "mem-1"


@pytest.mark.asyncio
async def test_remember_executor_routes_tagged_category():
    aiia = _StubAIIA()
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="remember")

    await executor.execute(_msg("decisions: switched routing to JSON path on E4B"))

    assert aiia.remember_calls[0]["category"] == "decisions"
    assert aiia.remember_calls[0]["fact"] == "switched routing to JSON path on E4B"


@pytest.mark.asyncio
async def test_remember_executor_rejects_empty_text():
    aiia = _StubAIIA()
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="remember")

    with pytest.raises(ValueError):
        await executor.execute(_msg("   "))


@pytest.mark.asyncio
async def test_remember_executor_raises_when_aiia_missing():
    async def _none_getter():
        return None

    executor = AIIAMemoryExecutor(_none_getter, mode="remember")

    with pytest.raises(RuntimeError, match="AIIA is not available"):
        await executor.execute(_msg("hi"))


# ─── AIIAMemoryExecutor — search mode ───────────────────────────────


@pytest.mark.asyncio
async def test_search_executor_returns_formatted_results():
    knowledge = _StubKnowledge(
        results=[
            {
                "source": "docs/auth.md",
                "text": "Auth uses JWT",
                "relevance_score": 0.91,
            },
            {"source": "routes/auth.py", "text": "def login(", "relevance_score": 0.85},
        ]
    )
    aiia = _StubAIIA(knowledge=knowledge)
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="search", n_results=2)

    result = await executor.execute(_msg("how does auth work"))

    assert knowledge.calls == [("how does auth work", 2)]
    assert result.artifact_name == "search-results"
    text = result.parts[0].text
    assert "[1] docs/auth.md (score: 0.91)" in text
    assert "Auth uses JWT" in text
    assert "[2] routes/auth.py (score: 0.85)" in text
    assert result.metadata["n_results"] == 2


@pytest.mark.asyncio
async def test_search_executor_handles_empty_results():
    knowledge = _StubKnowledge(results=[])
    aiia = _StubAIIA(knowledge=knowledge)
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="search")

    result = await executor.execute(_msg("nonexistent thing"))

    assert "No matching documents found." in result.parts[0].text
    assert result.metadata["n_results"] == 0


@pytest.mark.asyncio
async def test_search_executor_raises_when_aiia_lacks_knowledge_store():
    aiia = _StubAIIA(knowledge=None)
    getter = await _stub_getter_factory(aiia)
    executor = AIIAMemoryExecutor(getter, mode="search")

    with pytest.raises(RuntimeError, match="no knowledge store available"):
        await executor.execute(_msg("query"))


def test_invalid_mode_raises_at_construction():
    async def _g():
        return _StubAIIA()

    with pytest.raises(ValueError, match="mode must be remember|search"):
        AIIAMemoryExecutor(_g, mode="delete")  # type: ignore[arg-type]


# ─── bootstrap registration ─────────────────────────────────────────


def test_bootstrap_registers_all_dev_tools_without_aiia(monkeypatch):
    """Without aiia_getter, the 4 dev-tool agents (brain-status, run-tests,
    review-security, generate-report) all register; the 3 AIIA agents skip."""
    # Force the brain-CLI lookup to succeed without depending on host PATH
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: "/fake/bin/brain" if name == "brain" else None,
    )

    registry = register_default_agents(base_url="http://test:8100", aiia_getter=None)

    expected_dev = {
        "brain-status",
        "run-tests",
        "review-security",
        "generate-report",
    }
    assert {a.agent_id for a in registry.all()} == expected_dev


@pytest.mark.asyncio
async def test_bootstrap_registers_aiia_agents_when_getter_provided(monkeypatch):
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: "/fake/bin/brain" if name == "brain" else None,
    )

    aiia = _StubAIIA(knowledge=_StubKnowledge([]))

    async def _getter():
        return aiia

    registry = register_default_agents(
        base_url="http://test:8100",
        aiia_getter=_getter,
    )

    expected = {
        "brain-status",
        "run-tests",
        "review-security",
        "generate-report",
        "aiia-ask",
        "aiia-remember",
        "aiia-search",
        "aiia-status",
        "aiia-log-story",
        "aiia-educate",
    }
    assert {a.agent_id for a in registry.all()} == expected


def test_bootstrap_skips_brain_cli_agents_when_brain_missing(monkeypatch, tmp_path):
    """If `brain` is not on PATH AND the fallback path doesn't exist,
    brain-status, review-security, and generate-report all skip but
    run-tests still registers (it uses sys.executable, no brain CLI needed).
    """
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: None,
    )
    # Isolate Path.home() to a tmp dir so the ~/aplora-local-brain/brain
    # fallback also fails — otherwise the real Mini's brain CLI is picked up.
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.Path.home",
        classmethod(lambda cls: tmp_path),
    )

    registry = register_default_agents(base_url="http://test:8100", aiia_getter=None)

    assert {a.agent_id for a in registry.all()} == {"run-tests"}


def test_dev_tool_agents_carry_expected_tags(monkeypatch):
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: "/fake/bin/brain" if name == "brain" else None,
    )

    registry = register_default_agents(base_url="http://test:8100", aiia_getter=None)

    for agent in registry.all():
        tags = set(agent.card.all_tags())
        assert "scope:global" in tags, f"{agent.agent_id} missing scope:global"
        assert "layer:dev-tools" in tags, f"{agent.agent_id} missing layer:dev-tools"


def test_tag_query_isolates_dev_tools_from_aiia(monkeypatch):
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: "/fake/bin/brain" if name == "brain" else None,
    )

    async def _getter():
        return _StubAIIA(knowledge=_StubKnowledge([]))

    registry = register_default_agents(base_url="http://test:8100", aiia_getter=_getter)

    dev_only = {a.agent_id for a in registry.query(tags=["layer:dev-tools"])}
    aiia_only = {a.agent_id for a in registry.query(tags=["layer:aiia"])}

    assert dev_only == {
        "brain-status",
        "run-tests",
        "review-security",
        "generate-report",
    }
    assert aiia_only == {
        "aiia-ask",
        "aiia-remember",
        "aiia-search",
        "aiia-status",
        "aiia-log-story",
        "aiia-educate",
    }
    assert dev_only.isdisjoint(aiia_only)


# ─── _find_brain_cli ────────────────────────────────────────────────


def test_find_brain_cli_uses_path_first(monkeypatch):
    """When `brain` is on PATH, return that path without checking the fallback."""
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: "/usr/local/bin/brain" if name == "brain" else None,
    )
    assert _find_brain_cli() == "/usr/local/bin/brain"


def test_find_brain_cli_falls_back_to_aplora_local_brain(monkeypatch, tmp_path):
    """When PATH lookup fails, check ~/aplora-local-brain/brain.

    This is the launchd case: the brain service starts with a minimal PATH
    that doesn't include the user's home, but the CLI is at a known absolute
    location. Without this fallback the brain-status, review-security, and
    generate-report agents silently skip registration.
    """
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    brain = tmp_path / "aplora-local-brain" / "brain"
    brain.parent.mkdir(parents=True)
    brain.write_text("#!/bin/bash\necho ok\n")
    brain.chmod(0o755)

    assert _find_brain_cli() == str(brain)


def test_find_brain_cli_returns_none_when_neither_works(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    # No brain CLI created — fallback path doesn't exist.
    assert _find_brain_cli() is None


def test_find_brain_cli_skips_non_executable_fallback(monkeypatch, tmp_path):
    """A file at the fallback path that isn't executable should be ignored."""
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    brain = tmp_path / "aplora-local-brain" / "brain"
    brain.parent.mkdir(parents=True)
    brain.write_text("not executable")
    brain.chmod(0o644)

    assert _find_brain_cli() is None


def test_bootstrap_uses_brain_cli_fallback(monkeypatch, tmp_path):
    """End-to-end: when only the fallback path works, all 4 dev-tools register."""
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "local_brain.a2a.bootstrap.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    brain = tmp_path / "aplora-local-brain" / "brain"
    brain.parent.mkdir(parents=True)
    brain.write_text("#!/bin/bash\necho ok\n")
    brain.chmod(0o755)

    registry = register_default_agents(base_url="http://test:8100", aiia_getter=None)

    assert {a.agent_id for a in registry.all()} == {
        "brain-status",
        "run-tests",
        "review-security",
        "generate-report",
    }
