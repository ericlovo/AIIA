"""Offline tests for the A2A EducationExecutor.

A fake AIIA captures the question + teaching `context` and the requested
n_results, so we can assert level parsing and grounding without a live
brain or LLM.
"""

from __future__ import annotations

import pytest

from local_brain.a2a.executors.education_executor import (
    DEFAULT_LEVEL,
    LEVEL_DIRECTIVES,
    EducationExecutor,
)
from local_brain.a2a.schema import Message, TextPart


class FakeAIIA:
    def __init__(self, answer: str = "An explanation."):
        self.answer = answer
        self.calls: list[dict] = []

    async def ask(self, question, context=None, include_sessions=True, n_results=5, **kw):
        self.calls.append({"question": question, "context": context, "n_results": n_results})
        return {"answer": self.answer, "sources": ["doc1"], "model": "local"}


def _msg(text: str) -> Message:
    return Message(role="user", parts=[TextPart(text=text)])


def _getter(aiia):
    async def _get():
        return aiia

    return _get


class TestLevelParsing:
    @pytest.mark.asyncio
    async def test_default_level_when_no_prefix(self):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia))
        result = await ex.execute(_msg("What is iambic pentameter?"))
        assert result.metadata["level"] == DEFAULT_LEVEL
        # The directive for the default level is injected as teaching context.
        assert LEVEL_DIRECTIVES[DEFAULT_LEVEL] in aiia.calls[0]["context"]
        assert aiia.calls[0]["question"] == "What is iambic pentameter?"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("level", sorted(LEVEL_DIRECTIVES))
    async def test_each_level_prefix_is_honored(self, level):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia))
        result = await ex.execute(_msg(f"{level}: Explain Mrs Dalloway."))
        assert result.metadata["level"] == level
        assert aiia.calls[0]["question"] == "Explain Mrs Dalloway."
        assert LEVEL_DIRECTIVES[level] in aiia.calls[0]["context"]

    @pytest.mark.asyncio
    async def test_unknown_prefix_is_kept_in_question(self):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia))
        await ex.execute(_msg("note: this is part of the question"))
        # 'note' isn't a level, so nothing is stripped.
        assert aiia.calls[0]["question"] == "note: this is part of the question"

    @pytest.mark.asyncio
    async def test_case_insensitive_prefix(self):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia))
        result = await ex.execute(_msg("ELI5: what is a sonnet?"))
        assert result.metadata["level"] == "eli5"


class TestGroundingAndOutput:
    @pytest.mark.asyncio
    async def test_answer_and_metadata_passed_through(self):
        aiia = FakeAIIA(answer="A sonnet is a 14-line poem.")
        ex = EducationExecutor(_getter(aiia))
        result = await ex.execute(_msg("eli5: what is a sonnet?"))
        assert result.parts[0].text == "A sonnet is a 14-line poem."
        assert result.metadata["level"] == "eli5"
        assert result.metadata["sources"] == ["doc1"]  # grounding metadata survives
        assert result.artifact_name == "education-response"

    @pytest.mark.asyncio
    async def test_teaching_frame_demands_grounding(self):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia))
        await ex.execute(_msg("What is a metaphor?"))
        ctx = aiia.calls[0]["context"].lower()
        assert "knowledge" in ctx and "invent" in ctx

    @pytest.mark.asyncio
    async def test_custom_n_results(self):
        aiia = FakeAIIA()
        ex = EducationExecutor(_getter(aiia), n_results=3)
        await ex.execute(_msg("What is a metaphor?"))
        assert aiia.calls[0]["n_results"] == 3


class TestGuards:
    @pytest.mark.asyncio
    async def test_empty_message_rejected(self):
        ex = EducationExecutor(_getter(FakeAIIA()))
        with pytest.raises(ValueError, match="non-empty"):
            await ex.execute(_msg("   "))

    @pytest.mark.asyncio
    async def test_missing_aiia_raises(self):
        ex = EducationExecutor(_getter(None))
        with pytest.raises(RuntimeError, match="not available"):
            await ex.execute(_msg("What is a metaphor?"))

    def test_invalid_default_level_rejected(self):
        with pytest.raises(ValueError, match="default_level"):
            EducationExecutor(_getter(FakeAIIA()), default_level="phd")


class TestRegistration:
    def test_educate_agent_is_registered(self):
        from local_brain.a2a.bootstrap import register_default_agents

        async def _getter_fn():
            return FakeAIIA()

        registry = register_default_agents(aiia_getter=_getter_fn)
        agent = registry.get("aiia-educate")
        assert agent is not None
