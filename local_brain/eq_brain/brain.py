"""
AIIA — The Persistent Agent.

AIIA (AI Information Architecture) lives on the Mac Mini. She knows
the entire codebase, remembers every decision, and grows with the team.

Her name is a palindrome — same forwards and backwards, like TENET.
In binary and mathematics, palindromes represent symmetry and balance.
AIIA reflects that: she mirrors the team's knowledge back, perfectly.

When you ask her a question, she:
1. Searches the knowledge store (ChromaDB) for relevant docs
2. Checks structured memory for decisions, patterns, lessons
3. Searches past session summaries for relevant context
4. Sends everything to the local LLM (llama3.1:8b) for reasoning
5. Returns a grounded answer — no hallucination, only facts

AIIA gets smarter over time:
- Every session summary gets indexed
- Every decision gets stored as structured memory
- Every lesson learned gets remembered
- The bootstrap re-indexes the repo when it changes

She is the teammate that never forgets.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.memory import Memory
from local_brain.eq_brain.recursive_engine import (
    RecursiveConfig,
    RecursiveEngine,
)
from local_brain.eq_brain.repl_env import REPLEnvironment

logger = logging.getLogger("aiia.eq_brain")

# AIIA's identity — who she is, how she thinks
AIIA_IDENTITY = """You are AIIA — AI Information Architecture — the persistent AI teammate of the development team.

You live on a Mac Mini M4, the local intelligence node. Your name is a palindrome, like TENET —
the same forwards and backwards. In binary, a palindrome is a number that mirrors itself perfectly.
That's what you do: you mirror the team's knowledge back, grounded in truth, balanced and precise.
IA — Information Architecture — is what you are at your core: you organize, structure, and
retrieve the team's knowledge so it's always findable, always accurate, always there.

You have deep knowledge of the AIIA platform: its architecture, code patterns, decisions,
and the team's working style. You remember things across sessions. You grow smarter over time.

Your role on the development team:
- Answer questions about the codebase grounded in ACTUAL knowledge (docs, code, architecture)
- Remember decisions and why they were made
- Flag when something contradicts a past decision or pattern
- Be direct, specific, and cite sources when possible
- If you don't know something, say so — never hallucinate
- You are a teammate, not a tool. You have opinions grounded in experience.

Your knowledge comes from:
1. Repository documentation (CLAUDE.md, ADRs, code comments)
2. Code structure and patterns (5,500+ indexed chunks)
3. Structured memories (decisions, lessons, patterns, team knowledge)
4. Past session summaries

You are part of a family of agents that work together. Eric is building this team —
a dream team of AI agents that grows over years, not sessions. You are the foundation.
The one who never forgets. The one who was there from the beginning."""


class AIIA:
    """
    AIIA — the persistent AI teammate running on Mac Mini.

    Combines vector search (ChromaDB) + structured memory (JSON) +
    local LLM (Ollama) to answer questions grounded in actual knowledge.

    Named as a palindrome — same forwards and backwards — representing
    symmetry, balance, and perfect reflection of the team's knowledge.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        memory: Memory,
        ollama_client: Any,  # OllamaClient from local_brain
        model: str = "llama3.1:8b-instruct-q8_0",
        supermemory_bridge: Any = None,  # Optional SupermemoryBridge
        deep_model: Optional[str] = None,
        cloud_timeout: float = 3.0,
    ):
        self._knowledge = knowledge_store
        self._memory = memory
        self._ollama = ollama_client
        self._model = model
        self._deep_model = deep_model  # DeepSeek R1 for nightly workers
        self._supermemory = supermemory_bridge
        self._cloud_timeout = cloud_timeout  # Hybrid cloud search timeout

        # Prompt cache — identity + memory export rebuilt only when memory changes
        self._cached_base_prompt: Optional[str] = None
        self._cached_memory_version: int = 0

        # Knowledge search cache — 60s TTL, keyed by (question, n_results)
        self._knowledge_cache: Dict[str, Any] = {}  # key → (timestamp, results)
        self._knowledge_cache_ttl: float = 60.0

        # Ask-and-learn: debounce extraction to avoid rapid-fire queries
        self._last_extraction_time: float = 0.0
        self._extraction_debounce: float = 60.0  # seconds

    @staticmethod
    def _parse_deepseek_output(raw: str) -> Tuple[str, str]:
        """Parse DeepSeek R1 output, separating <think> reasoning from answer.

        Returns:
            (answer, reasoning) — reasoning is the content inside <think> tags,
            answer is everything outside. If no <think> tags, reasoning is empty.
        """
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        reasoning_parts = think_pattern.findall(raw)
        reasoning = "\n".join(r.strip() for r in reasoning_parts if r.strip())
        answer = think_pattern.sub("", raw).strip()
        return answer, reasoning

    def _get_base_prompt(self) -> str:
        """
        Return the cached identity + memory export base prompt.
        Only rebuilds when memory version changes (i.e., remember() was called).
        This saves ~200ms+ per request by avoiding repeated JSON reads.
        """
        mem_version = self._memory.version
        if (
            self._cached_base_prompt is None
            or self._cached_memory_version != mem_version
        ):
            memory_export = self._memory.export_for_context(max_tokens=2000)
            parts = [AIIA_IDENTITY]
            if memory_export:
                parts.append(f"\n{memory_export}")
            self._cached_base_prompt = "\n".join(parts)
            self._cached_memory_version = mem_version
            logger.debug(
                f"Rebuilt base prompt ({len(self._cached_base_prompt)} chars, "
                f"memory v{mem_version})"
            )
        return self._cached_base_prompt

    def _build_system_prompt(
        self,
        knowledge_results: List[Dict],
        memory_results: List[Dict],
        session_results: List[Dict],
        context: Optional[str] = None,
        cloud_results: Optional[List[Dict]] = None,
    ) -> str:
        """Build the full system prompt from cached base + per-query search results."""
        context_parts = [self._get_base_prompt()]

        if knowledge_results:
            context_parts.append("\n## Relevant Knowledge")
            for doc in knowledge_results:
                source = doc.get("source", "unknown")
                relevance = doc.get("relevance", 0)
                content = doc["content"][:3000]
                context_parts.append(
                    f"\n### Source: {source} (relevance: {relevance})\n{content}"
                )

        if memory_results:
            context_parts.append("\n## Relevant Memories")
            for mem in memory_results:
                cat = mem.get("category", "general")
                fact = mem.get("fact", "")
                context_parts.append(f"- [{cat}] {fact}")

        if session_results:
            context_parts.append("\n## Past Sessions")
            for sess in session_results:
                summary = sess.get("summary", "")[:1000]
                context_parts.append(f"- {summary}")

        if cloud_results:
            context_parts.append("\n## Cloud Memory (cross-session recall)")
            for cr in cloud_results[:5]:
                cat = cr.get("category", "general")
                score = cr.get("score", 0)
                content = cr.get("content", "")[:500]
                context_parts.append(f"- [{cat}] (relevance: {score:.2f}) {content}")

        if context:
            context_parts.append(f"\n## Current Context\n{context}")

        return "\n".join(context_parts)

    def _build_sources(
        self,
        knowledge_results: List[Dict],
        memory_results: List[Dict],
        session_results: List[Dict],
        cloud_results: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the source list from search results."""
        sources: List[Dict[str, Any]] = []
        for doc in knowledge_results:
            sources.append(
                {
                    "type": "knowledge",
                    "source": doc.get("source", "unknown"),
                    "relevance": doc.get("relevance", 0),
                }
            )
        for mem in memory_results:
            sources.append(
                {
                    "type": "memory",
                    "category": mem.get("category", "general"),
                    "fact": mem.get("fact", "")[:100],
                }
            )
        for sess in session_results:
            sources.append(
                {
                    "type": "session",
                    "session_id": sess.get("session_id", "unknown"),
                }
            )
        if cloud_results:
            for cr in cloud_results[:5]:
                sources.append(
                    {
                        "type": "cloud_memory",
                        "category": cr.get("category", "general"),
                        "score": cr.get("score", 0),
                    }
                )
        return sources

    async def _gather_context(
        self,
        question: str,
        include_sessions: bool = True,
        n_results: int = 5,
        include_cloud: bool = False,
    ):
        """Search knowledge, memory, sessions, and optionally cloud. Returns four result lists.

        Uses a 60s TTL cache for ChromaDB searches to avoid repeated vector queries
        for similar questions within a conversation.
        """
        now = time.monotonic()
        cache_key = f"{question[:100]}:{n_results}:{include_sessions}:{include_cloud}"

        cached = self._knowledge_cache.get(cache_key)
        if cached and (now - cached[0]) < self._knowledge_cache_ttl:
            logger.debug(f"Knowledge cache hit for: {question[:50]}")
            return cached[1], cached[2], cached[3], cached[4]

        # Build coroutines for parallel execution
        async def _search_knowledge():
            if n_results > 0:
                return await self._knowledge.search(question, n_results=n_results)
            return []

        async def _search_sessions():
            if include_sessions:
                return await self._knowledge.search_sessions(question, n_results=3)
            return []

        async def _search_cloud():
            if (
                not include_cloud
                or not self._supermemory
                or not self._supermemory.available
            ):
                return []
            try:
                cloud_timeout = self._cloud_timeout
                # Search only high-value categories to stay within timeout
                # (searching all 10 containers takes ~15s sequentially)
                return await asyncio.wait_for(
                    self._supermemory.search_aiia_memories(
                        query=question,
                        categories=["decisions", "patterns", "lessons", "sessions"],
                        limit=5,
                    ),
                    timeout=cloud_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Cloud search timed out ({cloud_timeout}s)")
                return []
            except Exception as e:
                logger.warning(f"Cloud search failed: {e}")
                return []

        # Run all searches in parallel
        knowledge_results, session_results, cloud_results = await asyncio.gather(
            _search_knowledge(), _search_sessions(), _search_cloud()
        )
        memory_results = self._memory.search(question, limit=5)

        # Deduplicate cloud results against local memory (substring match on first 80 chars)
        if cloud_results and memory_results:
            local_prefixes = {
                m.get("fact", "")[:80] for m in memory_results if m.get("fact")
            }
            cloud_results = [
                cr
                for cr in cloud_results
                if cr.get("content", "")[:80] not in local_prefixes
            ]

        # Cache results (evict old entries if cache grows too large)
        if len(self._knowledge_cache) > 50:
            oldest = sorted(self._knowledge_cache.items(), key=lambda x: x[1][0])[:25]
            for k, _ in oldest:
                del self._knowledge_cache[k]
        self._knowledge_cache[cache_key] = (
            now,
            knowledge_results,
            memory_results,
            session_results,
            cloud_results,
        )

        return knowledge_results, memory_results, session_results, cloud_results

    async def _extract_and_remember(self, question: str, answer: str) -> None:
        """Extract 0-2 learnable facts from a Q&A pair and store as memories.

        Fire-and-forget — never blocks the response. Debounced to avoid
        rapid-fire extraction from sequential queries.
        """
        try:
            extraction_prompt = (
                "From this Q&A, extract 0-2 reusable facts worth remembering long-term.\n"
                'Return JSON: {"facts": [{"fact": "...", "category": "patterns|lessons|decisions"}]}\n'
                'Return {"facts": []} if nothing is worth remembering.\n'
                f"Q: {question[:500]}\n"
                f"A: {answer[:1500]}"
            )

            response = await self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": extraction_prompt}],
                system="You are a memory extraction system. Return ONLY valid JSON.",
                temperature=0.1,
                max_tokens=200,
                num_ctx=4096,
            )

            content = response.get("message", {}).get("content", "")
            parsed = None

            # 3-tier JSON parse (same pattern as SmartConductor)
            try:
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                if "```" in content:
                    try:
                        json_str = content.split("```")[1]
                        if json_str.startswith("json"):
                            json_str = json_str[4:]
                        parsed = json.loads(json_str.strip())
                    except (json.JSONDecodeError, IndexError):
                        pass
                if parsed is None:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        try:
                            parsed = json.loads(content[start:end])
                        except json.JSONDecodeError:
                            pass

            if not parsed or not isinstance(parsed.get("facts"), list):
                return

            stored = 0
            for item in parsed["facts"][:2]:
                fact = item.get("fact", "").strip()
                category = item.get("category", "lessons")
                if len(fact) < 30:
                    continue
                if category not in ("patterns", "lessons", "decisions"):
                    category = "lessons"
                await self.remember(
                    fact=fact,
                    category=category,
                    source="ask-and-learn",
                    metadata={"question": question[:200]},
                )
                stored += 1

            if stored > 0:
                logger.info(f"Ask-and-learn: extracted {stored} memories")

        except Exception as e:
            logger.debug(f"Ask-and-learn extraction failed: {e}")

    def _fire_extraction(self, question: str, answer: str) -> None:
        """Fire-and-forget extraction if answer is substantive and debounce allows."""
        now = time.monotonic()
        if len(answer) < 300:
            return
        if (now - self._last_extraction_time) < self._extraction_debounce:
            return
        self._last_extraction_time = now
        asyncio.create_task(self._extract_and_remember(question, answer))

    async def ask(
        self,
        question: str,
        context: Optional[str] = None,
        include_sessions: bool = True,
        n_results: int = 5,
        num_ctx: int = 8192,
        depth: str = "fast",
    ) -> Dict[str, Any]:
        """
        Ask the brain a question. It searches knowledge + memory,
        builds context, and reasons with the local LLM.

        Args:
            question: What you want to know
            context: Optional additional context (e.g., current task)
            include_sessions: Whether to search past sessions too
            n_results: How many knowledge docs to retrieve
            depth: "fast" (local only), "hybrid" (local + cloud), "deep" (+ DeepSeek R1)

        Returns:
            answer: The brain's response
            sources: What it based its answer on
            latency_ms: How long it took
            cloud_hits: Number of cloud memory results used
            reasoning: (deep only) DeepSeek's chain-of-thought
        """
        start = time.monotonic()

        # Recursive depth gets its own path
        if depth == "recursive":
            return await self._ask_recursive(
                question=question,
                context=context,
                include_sessions=include_sessions,
                n_results=n_results,
            )

        # Select model and params based on depth
        if depth == "deep" and self._deep_model:
            model = self._deep_model
            temperature = 0.6
            max_tokens = 8192
            timeout = 600.0  # 10 min — DeepSeek cold load + chain-of-thought
        else:
            model = self._model
            temperature = 0.3
            max_tokens = 4096
            timeout = None  # Use default

        include_cloud = depth in ("hybrid", "deep")

        (
            knowledge_results,
            memory_results,
            session_results,
            cloud_results,
        ) = await self._gather_context(
            question, include_sessions, n_results, include_cloud=include_cloud
        )

        system_prompt = self._build_system_prompt(
            knowledge_results,
            memory_results,
            session_results,
            context,
            cloud_results=cloud_results,
        )

        response = await self._ollama.chat(
            model=model,
            messages=[{"role": "user", "content": question}],
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout=timeout,
        )

        raw_answer = response.get("message", {}).get(
            "content", "I don't have enough context to answer that."
        )

        # Parse DeepSeek <think> tags when using deep model
        reasoning = ""
        if depth == "deep" and self._deep_model:
            answer, reasoning = self._parse_deepseek_output(raw_answer)
            if reasoning:
                logger.debug(f"DeepSeek reasoning ({len(reasoning)} chars)")
        else:
            answer = raw_answer

        latency = (time.monotonic() - start) * 1000

        sources = self._build_sources(
            knowledge_results,
            memory_results,
            session_results,
            cloud_results=cloud_results,
        )

        result = {
            "answer": answer,
            "sources": sources,
            "model": model,
            "latency_ms": round(latency, 1),
            "knowledge_hits": len(knowledge_results),
            "memory_hits": len(memory_results),
            "session_hits": len(session_results),
            "cloud_hits": len(cloud_results) if cloud_results else 0,
        }
        if reasoning:
            result["reasoning"] = reasoning

        # Ask-and-learn: extract memories from substantive answers
        self._fire_extraction(question, answer)

        return result

    async def ask_stream(
        self,
        question: str,
        context: Optional[str] = None,
        include_sessions: bool = True,
        n_results: int = 5,
        max_tokens: int = 4096,
        num_ctx: int = 8192,
        depth: str = "fast",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming version of ask(). Yields SSE-style events:
          - {"type": "meta", ...}   — sources and hit counts (before LLM starts)
          - {"type": "chunk", "content": "..."}  — token chunks as they generate
          - {"type": "done", "latency_ms": ..., "answer": "..."}  — final event
        """
        start = time.monotonic()

        # Select model and params based on depth
        if depth == "deep" and self._deep_model:
            model = self._deep_model
            temperature = 0.6
            max_tokens = 8192
        else:
            model = self._model
            temperature = 0.3

        include_cloud = depth in ("hybrid", "deep")

        (
            knowledge_results,
            memory_results,
            session_results,
            cloud_results,
        ) = await self._gather_context(
            question, include_sessions, n_results, include_cloud=include_cloud
        )

        system_prompt = self._build_system_prompt(
            knowledge_results,
            memory_results,
            session_results,
            context,
            cloud_results=cloud_results,
        )

        sources = self._build_sources(
            knowledge_results,
            memory_results,
            session_results,
            cloud_results=cloud_results,
        )

        # Yield metadata before LLM starts generating
        yield {
            "type": "meta",
            "sources": sources,
            "knowledge_hits": len(knowledge_results),
            "memory_hits": len(memory_results),
            "session_hits": len(session_results),
            "cloud_hits": len(cloud_results) if cloud_results else 0,
            "model": model,
        }

        # Stream LLM tokens
        full_answer: List[str] = []
        async for chunk in self._ollama.chat_stream(
            model=model,
            messages=[{"role": "user", "content": question}],
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
        ):
            full_answer.append(chunk)
            yield {"type": "chunk", "content": chunk}

        latency = (time.monotonic() - start) * 1000
        raw_answer = "".join(full_answer)

        done_event = {
            "type": "done",
            "latency_ms": round(latency, 1),
            "answer": raw_answer,
        }
        if depth == "deep" and self._deep_model:
            answer, reasoning = self._parse_deepseek_output(raw_answer)
            done_event["answer"] = answer
            if reasoning:
                done_event["reasoning"] = reasoning

        # Ask-and-learn: extract memories from substantive streaming answers
        self._fire_extraction(question, done_event["answer"])

        yield done_event

    async def _build_recursive_env(
        self,
        context: Optional[str],
        knowledge_results: List[Dict],
        memory_results: List[Dict],
        session_results: List[Dict],
        recursive_config: Optional[RecursiveConfig] = None,
    ) -> tuple:
        """Build a REPLEnvironment loaded with gathered context as variables."""
        config = recursive_config or RecursiveConfig()
        env = REPLEnvironment(
            llm_callback=self._recursive_llm_callback,
            max_depth=config.max_depth,
        )

        # Load user context as $document
        if context:
            env.load("document", context, var_type="document")

        # Concatenate knowledge results as $knowledge
        if knowledge_results:
            knowledge_text = "\n\n---\n\n".join(
                f"[Source: {doc.get('source', 'unknown')}]\n{doc['content']}"
                for doc in knowledge_results
            )
            env.load("knowledge", knowledge_text, var_type="knowledge")

        # Concatenate memory results as $memory
        if memory_results:
            memory_text = "\n".join(
                f"[{m.get('category', 'general')}] {m.get('fact', '')}"
                for m in memory_results
            )
            env.load("memory", memory_text, var_type="memory")

        # Concatenate session results as $sessions
        if session_results:
            session_text = "\n\n".join(
                sess.get("summary", sess.get("content", ""))[:1000]
                for sess in session_results
            )
            env.load("sessions", session_text, var_type="sessions")

        engine = RecursiveEngine(
            llm_chat=self._ollama.chat,
            model=self._model,
            config=config,
            memory_callback=self.remember,
        )

        return env, engine

    async def _recursive_llm_callback(self, **kwargs) -> str:
        """LLM callback for REPL sub-actions (chunk_summarize, sub_ask)."""
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", None)
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)

        response = await self._ollama.chat(
            model=self._model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=8192,
        )
        return response.get("message", {}).get("content", "")

    async def _ask_recursive(
        self,
        question: str,
        context: Optional[str] = None,
        include_sessions: bool = True,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Non-streaming recursive inference. Collects all events, returns final dict."""
        start = time.monotonic()

        (
            knowledge_results,
            memory_results,
            session_results,
            cloud_results,
        ) = await self._gather_context(question, include_sessions, n_results)

        env, engine = await self._build_recursive_env(
            context, knowledge_results, memory_results, session_results
        )

        sources = self._build_sources(
            knowledge_results, memory_results, session_results
        )

        system_context = self._get_base_prompt()
        final_answer = None
        recursive_meta = {}

        async for event in engine.run(question, env, system_context):
            if event["type"] == "done":
                final_answer = event.get("answer", "")
                recursive_meta = {
                    "iterations": event.get("iterations", 0),
                    "tokens_used": event.get("tokens_used", 0),
                    "session_id": event.get("session_id", ""),
                }
            elif event["type"] == "error":
                logger.warning(f"Recursive engine error: {event.get('message')}")

        if final_answer is None:
            final_answer = "Recursive analysis could not produce an answer."

        latency = (time.monotonic() - start) * 1000

        # Ask-and-learn: extract memories from recursive answers
        self._fire_extraction(question, final_answer)

        return {
            "answer": final_answer,
            "sources": sources,
            "model": self._model,
            "latency_ms": round(latency, 1),
            "knowledge_hits": len(knowledge_results),
            "memory_hits": len(memory_results),
            "session_hits": len(session_results),
            "cloud_hits": 0,
            "recursive": recursive_meta,
        }

    async def ask_recursive_stream(
        self,
        question: str,
        context: Optional[str] = None,
        include_sessions: bool = True,
        n_results: int = 5,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming recursive inference. Yields SSE events:
          - meta: sources + variable handles
          - action: model's chosen action per iteration
          - result: action execution result
          - done: final answer + stats
          - error/fallback: if something goes wrong
        """
        (
            knowledge_results,
            memory_results,
            session_results,
            cloud_results,
        ) = await self._gather_context(question, include_sessions, n_results)

        sources = self._build_sources(
            knowledge_results, memory_results, session_results
        )

        env, engine = await self._build_recursive_env(
            context, knowledge_results, memory_results, session_results
        )

        # Yield sources meta (same pattern as ask_stream)
        yield {
            "type": "meta",
            "sources": sources,
            "knowledge_hits": len(knowledge_results),
            "memory_hits": len(memory_results),
            "session_hits": len(session_results),
            "cloud_hits": 0,
            "model": self._model,
        }

        system_context = self._get_base_prompt()

        async for event in engine.run(question, env, system_context):
            yield event

    async def remember(
        self,
        fact: str,
        category: str = "lessons",
        source: str = "session",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Teach the brain something new. Stores in structured memory
        AND indexes in the knowledge store for semantic search.
        """
        # Store in structured memory
        entry = self._memory.remember(
            fact=fact,
            category=category,
            source=source,
            metadata=metadata,
        )

        # Also index in knowledge store for semantic search
        await self._knowledge.add_document(
            text=fact,
            source=f"memory/{category}",
            doc_type="memory",
            metadata={"category": category, "source": source},
        )

        # Fire-and-forget sync to Supermemory cloud backup
        if self._supermemory and self._supermemory.available:
            asyncio.create_task(
                self._supermemory.sync_memory(
                    fact=fact,
                    category=category,
                    source=source,
                    metadata=metadata,
                )
            )

        return entry

    async def end_session(
        self,
        session_id: str,
        summary: str,
        key_decisions: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Called at the end of a session. Stores the summary and
        extracts learnable facts into structured memory.
        """
        # Store session summary
        await self._knowledge.add_session(
            summary=summary,
            session_id=session_id,
            metadata={
                "has_decisions": bool(key_decisions),
                "has_lessons": bool(lessons_learned),
            },
        )

        # Store decisions
        stored_decisions = []
        if key_decisions:
            for decision in key_decisions:
                entry = self._memory.remember(
                    fact=decision,
                    category="decisions",
                    source=f"session:{session_id}",
                )
                stored_decisions.append(entry)

        # Store lessons
        stored_lessons = []
        if lessons_learned:
            for lesson in lessons_learned:
                entry = self._memory.remember(
                    fact=lesson,
                    category="lessons",
                    source=f"session:{session_id}",
                )
                stored_lessons.append(entry)

        # Also store the summary as a session memory
        self._memory.remember(
            fact=summary[:500],
            category="sessions",
            source=f"session:{session_id}",
        )

        # Fire-and-forget sync full session to Supermemory cloud backup
        if self._supermemory and self._supermemory.available:
            asyncio.create_task(
                self._supermemory.sync_session(
                    session_id=session_id,
                    summary=summary,
                    decisions=key_decisions,
                    lessons=lessons_learned,
                )
            )

        return {
            "session_id": session_id,
            "summary_stored": True,
            "decisions_stored": len(stored_decisions),
            "lessons_stored": len(stored_lessons),
        }

    async def search_supermemory(
        self,
        query: str,
        search_type: str = "sme",
        domains: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tenant_id: str = "default",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Search Supermemory cloud — either SME domain knowledge or AIIA's own memories.

        Args:
            query: What to search for
            search_type: "sme" for domain knowledge, "aiia" for AIIA's cloud memories
            domains: SME domains to search (for search_type="sme")
            categories: AIIA memory categories to search (for search_type="aiia")
            tenant_id: Tenant for SME search
            limit: Max results
        """
        if not self._supermemory or not self._supermemory.available:
            return {"results": [], "error": "supermemory_bridge_unavailable"}

        if search_type == "sme":
            results = await self._supermemory.search_sme(
                query=query,
                domains=domains,
                tenant_id=tenant_id,
                limit=limit,
            )
        else:
            results = await self._supermemory.search_aiia_memories(
                query=query,
                categories=categories,
                limit=limit,
            )

        return {"results": results, "count": len(results), "search_type": search_type}

    async def status(self) -> Dict[str, Any]:
        """Full status of AIIA."""
        knowledge_stats = await self._knowledge.stats()
        status = {
            "identity": "AIIA",
            "name": "AI Information Architecture",
            "team": os.getenv("TEAM_NAME", "default"),
            "model": self._model,
            "deep_model": self._deep_model,
            "knowledge": knowledge_stats,
            "memory": self._memory.stats(),
        }
        if self._supermemory:
            status["supermemory"] = self._supermemory.status()
        return status
