"""
EducationExecutor — an A2A tutor that explains, grounded in AIIA's knowledge.

Like AIIAExecutor, this routes through AIIA.ask() so every explanation is
backed by the ChromaDB knowledge store (including any literature/erdos
research corpus the loop has built), structured memory, and the local LLM —
not a bare model call. The difference is pedagogy: the executor injects a
teaching directive as `context` so AIIA explains *at a chosen level* rather
than just answering.

The level is picked off a 'level: question' wire prefix (the same text-only
convention AIIAMemoryExecutor uses for categories), so the protocol stays
text-only while callers can still say "eli5: what is iambic pentameter?".
Unknown or missing prefixes fall back to the default level.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.schema import Message, TextPart

logger = logging.getLogger("aplora.a2a.education_executor")

AIIAGetter = Callable[[], Awaitable[Any]]

# Each level maps to a teaching directive handed to AIIA as `context`.
LEVEL_DIRECTIVES: dict[str, str] = {
    "eli5": (
        "Explain to a curious 10-year-old. Use plain words and a concrete "
        "analogy. Avoid jargon; if you must use a term, define it simply."
    ),
    "beginner": (
        "Explain to someone new to the subject. Assume no background, define "
        "every term, and build up from first principles with a worked example."
    ),
    "intermediate": (
        "Explain to a learner with some background. Connect to what they likely "
        "already know and focus on the ideas that are easy to get wrong."
    ),
    "undergrad": (
        "Explain at an undergraduate university level: rigorous but accessible, "
        "with definitions, a clear line of reasoning, and a concrete example."
    ),
    "advanced": (
        "Explain to an advanced student. Be precise and technical, state "
        "assumptions, and note subtleties, edge cases, and common misconceptions."
    ),
    "expert": (
        "Explain to a domain expert. Be concise and exact, use proper notation "
        "and terminology, and focus on the non-obvious points."
    ),
}

DEFAULT_LEVEL = "undergrad"

_TEACHING_FRAME = (
    "You are a tutor. Teach the answer rather than just stating it: lead with a "
    "direct answer, then explain the reasoning, then give one example. Ground "
    "every claim in the retrieved knowledge; if the knowledge base does not "
    "cover something, say so plainly instead of inventing it. {directive}"
)


class EducationExecutor(AgentExecutor):
    """Wraps AIIA as a level-aware teaching agent."""

    def __init__(
        self,
        aiia_getter: AIIAGetter,
        *,
        default_level: str = DEFAULT_LEVEL,
        n_results: int = 6,
        artifact_name: str = "education-response",
    ) -> None:
        if default_level not in LEVEL_DIRECTIVES:
            raise ValueError(
                f"default_level must be one of {sorted(LEVEL_DIRECTIVES)}, got {default_level}"
            )
        self._get_aiia = aiia_getter
        self._default_level = default_level
        self._n_results = n_results
        self._artifact_name = artifact_name

    async def execute(self, message: Message) -> ExecutorResult:
        text = _join_text_parts(message)
        if not text.strip():
            raise ValueError("EducationExecutor requires a non-empty text part in the message")

        level, question = _parse_level(text, default=self._default_level)
        directive = LEVEL_DIRECTIVES[level]
        teaching_context = _TEACHING_FRAME.format(directive=directive)

        aiia = await self._get_aiia()
        if aiia is None:
            raise RuntimeError("AIIA is not available on this host")

        logger.info("education task level=%s q=%.80s", level, question)
        result = await aiia.ask(
            question,
            context=teaching_context,
            include_sessions=False,
            n_results=self._n_results,
        )

        answer = _extract_answer(result)
        metadata = {"level": level}
        if isinstance(result, dict):
            for k, v in result.items():
                if k != "answer" and _is_jsonable(v):
                    metadata[k] = v

        return ExecutorResult(
            parts=[TextPart(text=answer)],
            artifact_name=self._artifact_name,
            artifact_description=f"Tutored explanation from AIIA (level: {level})",
            metadata=metadata,
        )


# ─── helpers ─────────────────────────────────────────────────────────


def _join_text_parts(message: Message) -> str:
    texts = []
    for part in message.parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _parse_level(text: str, *, default: str) -> tuple[str, str]:
    """Accept either a plain question or a 'level: question' prefixed form."""
    if ":" in text:
        head, _, rest = text.partition(":")
        head_norm = head.strip().lower()
        if head_norm in LEVEL_DIRECTIVES and rest.strip():
            return head_norm, rest.strip()
    return default, text.strip()


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
