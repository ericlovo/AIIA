"""
ResearchEngine — orchestrates one research session on a topic.

Each run:
  1. Loads topic state (synthesis, gaps, indexed sources)
  2. Pre-fetches unprocessed seeds (up to 3 per session)
  3. Runs the recursive loop with a research-oriented system prompt
  4. Persists synthesis + gaps + run metadata back to the topic
"""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.recursive_engine import RecursiveConfig, RecursiveEngine
from local_brain.ollama_client import OllamaClient
from local_brain.research.fetcher import fetch_url
from local_brain.research.profiles import get_profile
from local_brain.research.repl_env import ResearchREPLEnvironment
from local_brain.research.topic import ResearchTopic, TopicStore

logger = logging.getLogger("aiia.research.engine")

_SEEDS_PER_SESSION = 3


def _system_prompt(topic: ResearchTopic) -> str:
    profile = get_profile(topic.profile)
    gaps = "\n".join(f"- {g}" for g in topic.gaps) if topic.gaps else "None yet — discover them."
    indexed = (
        "\n".join(f"- {s}" for s in topic.sources_indexed) if topic.sources_indexed else "None yet."
    )
    return f"""You are AIIA conducting structured deep research.

TOPIC: {topic.title}
QUESTION: {topic.question}

OPEN GAPS (questions from previous sessions to address):
{gaps}

ALREADY INDEXED SOURCES (skip these):
{indexed}

YOUR GOAL THIS SESSION:
{profile.goal}

PRINCIPLES:
{profile.principles}"""


class ResearchEngine:
    def __init__(
        self,
        ollama: OllamaClient,
        knowledge: KnowledgeStore,
        topic_store: TopicStore,
        model: str = "llama3.1:8b-instruct-q8_0",
        config: RecursiveConfig | None = None,
    ):
        self._ollama = ollama
        self._knowledge = knowledge
        self._topic_store = topic_store
        self._model = model
        # More iterations + larger budget than default — research sessions are deeper
        self._config = config or RecursiveConfig(max_iterations=20, token_budget=80_000)

    async def run(self, topic: ResearchTopic) -> AsyncGenerator[dict[str, Any], None]:
        """Run one research session on a topic. Yields SSE events."""

        env = ResearchREPLEnvironment(
            topic=topic,
            topic_store=self._topic_store,
            knowledge=self._knowledge,
            llm_callback=self._llm_simple,
            max_depth=3,
        )

        # Load topic context as REPL variables
        env.load("topic_question", topic.question, var_type="knowledge")
        if topic.synthesis:
            env.load("synthesis", topic.synthesis, var_type="knowledge", readonly=False)
        if topic.gaps:
            env.load("open_gaps", "\n".join(f"- {g}" for g in topic.gaps), var_type="knowledge")

        # Pre-fetch unprocessed seeds (fire before the loop so they're ready as variables)
        unprocessed = [s for s in topic.seeds if s not in topic.sources_indexed]
        for i, seed in enumerate(unprocessed[:_SEEDS_PER_SESSION]):
            try:
                canonical_url, text = await fetch_url(seed)
                env.load(f"seed_{i + 1}", text, var_type="document")
                logger.info(f"Pre-loaded seed {i + 1}: {canonical_url} ({len(text)} chars)")
            except Exception as e:
                logger.warning(f"Seed fetch failed for {seed}: {e}")

        engine = RecursiveEngine(
            llm_chat=self._ollama.chat,
            model=self._model,
            config=self._config,
            data_dir=self._knowledge._data_dir,
        )

        topic.run_count += 1
        topic.last_run = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._topic_store.save(topic)

        async for event in engine.run(
            question=topic.question,
            env=env,
            system_context=_system_prompt(topic),
        ):
            if event.get("type") == "done":
                # If the model never called update_synthesis, use the final answer as first synthesis
                if not topic.synthesis:
                    topic.synthesis = event.get("answer", "")
                    self._topic_store.save(topic)
            yield event

    async def _llm_simple(self, **kwargs) -> str:
        response = await self._ollama.chat(
            model=self._model,
            messages=kwargs.get("messages", []),
            system=kwargs.get("system"),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            num_ctx=16384,
        )
        return response.get("message", {}).get("content", "")
