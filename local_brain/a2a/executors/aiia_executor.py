"""
AIIAExecutor — delegates A2A tasks to AIIA (the Local Brain persona).

Rather than hitting the raw Ollama API, this executor routes each task
through AIIA.ask(), which gives the invocation access to ChromaDB
knowledge, structured memory, and the configured local model. That makes
AIIA a first-class A2A agent instead of a thin LLM passthrough.

Because AIIA is lazy-initialized (ChromaDB + embeddings are heavy), this
executor takes an async getter rather than an AIIA instance. The getter
resolves at invocation time, which means the first A2A call can trigger
AIIA's initialization and subsequent calls reuse the warm singleton.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.schema import Message, TextPart

logger = logging.getLogger("aplora.a2a.aiia_executor")

AIIAGetter = Callable[[], Awaitable[Any]]


class AIIAExecutor(AgentExecutor):
    def __init__(
        self,
        aiia_getter: AIIAGetter,
        *,
        include_sessions: bool = True,
        n_results: int = 5,
        artifact_name: str = "aiia-response",
    ) -> None:
        self._get_aiia = aiia_getter
        self._include_sessions = include_sessions
        self._n_results = n_results
        self._artifact_name = artifact_name

    async def execute(self, message: Message) -> ExecutorResult:
        question = _join_text_parts(message)
        if not question.strip():
            raise ValueError("AIIAExecutor requires a non-empty text part in the message")

        aiia = await self._get_aiia()
        if aiia is None:
            raise RuntimeError("AIIA is not available on this host")

        logger.info("delegating A2A task to AIIA: %.80s...", question)
        result = await aiia.ask(
            question,
            context=None,
            include_sessions=self._include_sessions,
            n_results=self._n_results,
        )

        answer = _extract_answer(result)
        metadata = {
            k: v
            for k, v in (result.items() if isinstance(result, dict) else [])
            if k != "answer" and _is_jsonable(v)
        }

        return ExecutorResult(
            parts=[TextPart(text=answer)],
            artifact_name=self._artifact_name,
            artifact_description="Answer from AIIA (knowledge + memory + local LLM)",
            metadata=metadata,
        )


def _join_text_parts(message: Message) -> str:
    texts = []
    for part in message.parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _extract_answer(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("answer", "response", "content", "text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return str(result)


def _is_jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))
