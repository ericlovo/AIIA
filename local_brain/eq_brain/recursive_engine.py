"""
Recursive Inference Engine — REPL loop over AIIA's local LLM.

Implements the RLM paper's core insight: instead of stuffing everything
into the context window, store content as variables and let the model
explore via structured JSON actions. Small model + recursion handles
inputs 2 orders of magnitude beyond its context window.

Uses Llama 3.1:8b with JSON actions (not tool_use — Ollama doesn't
support it). Same JSON extraction pattern proven by SmartConductor.

Phase 4 of the Local Brain roadmap.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional

from local_brain.eq_brain.repl_env import REPLEnvironment

logger = logging.getLogger("aiia.eq_brain.recursive_engine")


@dataclass
class RecursiveConfig:
    """Configuration for the recursive engine. Overridable via env vars."""

    max_iterations: int = 15
    max_depth: int = 3
    token_budget: int = 50_000
    max_parse_failures: int = 2
    temperature: float = 0.15
    history_window: int = 6  # Keep last N messages (3 action-result pairs)

    def __post_init__(self):
        self.max_iterations = int(
            os.getenv("RECURSIVE_MAX_ITERATIONS", str(self.max_iterations))
        )
        self.max_depth = int(os.getenv("RECURSIVE_MAX_DEPTH", str(self.max_depth)))
        self.token_budget = int(
            os.getenv("RECURSIVE_TOKEN_BUDGET", str(self.token_budget))
        )


@dataclass
class TokenLedger:
    """Tracks cumulative token usage across iterations."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    budget: int = 50_000

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.total)

    @property
    def exhausted(self) -> bool:
        return self.total >= self.budget

    def record(self, prompt_eval_count: int, eval_count: int):
        self.prompt_tokens += prompt_eval_count
        self.completion_tokens += eval_count

    def status(self) -> str:
        return f"{self.total}/{self.budget} tokens used ({self.remaining} remaining)"


# LLM call type: async (model, messages, system, temperature, max_tokens, num_ctx) -> Dict
LLMChatCallback = Callable[..., Coroutine[Any, Any, Dict[str, Any]]]


def _parse_json_action(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM output. 3-tier fallback matching SmartConductor pattern.

    1. Direct json.loads()
    2. Extract from markdown code block
    3. Find {...} substring
    """
    content = content.strip()

    # Tier 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Tier 2: markdown code block
    if "```" in content:
        try:
            json_str = content.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # Tier 3: brace extraction
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    return None


class RecursiveEngine:
    """
    REPL loop engine for recursive inference.

    Yields SSE events as it runs:
        - meta: session info, variable handles
        - action: model's chosen action
        - result: action execution result
        - done: final answer + stats
        - error: if something goes wrong
        - fallback: if switching to chunked processing
    """

    def __init__(
        self,
        llm_chat: LLMChatCallback,
        model: str,
        config: Optional[RecursiveConfig] = None,
        data_dir: str = "~/.aiia/eq_data",
        memory_callback: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ):
        self._llm_chat = llm_chat
        self._model = model
        self._config = config or RecursiveConfig()
        self._data_dir = os.path.expanduser(data_dir)
        self._memory_callback = (
            memory_callback  # async (fact, category, source, metadata) -> None
        )

    def _build_system_prompt(
        self,
        question: str,
        handles: List[Dict[str, Any]],
        iteration: int,
        token_status: str,
        action_schema: str,
    ) -> str:
        """Build the system prompt for each REPL iteration."""
        handles_json = json.dumps(handles, indent=2)

        return f"""You are AIIA performing recursive document analysis.

QUESTION: {question}

VARIABLES (you cannot see their full content — use actions to explore):
{handles_json}

{action_schema}

ITERATION: {iteration + 1}/{self._config.max_iterations}
TOKEN BUDGET: {token_status}

RULES:
- Your response must be ONLY valid JSON — no explanation, no markdown outside the JSON
- Choose ONE action per response
- You MUST call final() before running out of iterations
- Be efficient: peek strategic sections, search for key terms, then synthesize
- If you have gathered enough information, call final() immediately"""

    async def _llm_callback_simple(self, **kwargs) -> str:
        """Simplified LLM callback for REPL sub-actions (chunk_summarize, sub_ask)."""
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", None)
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)

        response = await self._llm_chat(
            model=self._model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=8192,
        )
        return response.get("message", {}).get("content", "")

    async def run(
        self,
        question: str,
        env: REPLEnvironment,
        system_context: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the REPL loop. Yields SSE events.

        Args:
            question: The user's question
            env: Pre-loaded REPLEnvironment with variables
            system_context: Optional extra system context (identity, etc.)
        """
        session_id = str(uuid.uuid4())[:12]
        config = self._config
        ledger = TokenLedger(budget=config.token_budget)
        action_schema = REPLEnvironment.action_schema()

        # Trajectory for logging
        trajectory: List[Dict[str, Any]] = []
        session_start = time.monotonic()

        # Yield meta event
        yield {
            "type": "meta",
            "session_id": session_id,
            "model": self._model,
            "variables": env.handles(),
            "max_iterations": config.max_iterations,
            "token_budget": config.token_budget,
        }

        messages: List[Dict[str, str]] = []
        parse_failures = 0
        final_answer = None

        for iteration in range(config.max_iterations):
            if ledger.exhausted:
                yield {
                    "type": "error",
                    "message": f"Token budget exhausted: {ledger.status()}",
                    "iteration": iteration,
                }
                break

            # Build system prompt with current variable handles
            system_prompt = self._build_system_prompt(
                question=question,
                handles=env.handles(),
                iteration=iteration,
                token_status=ledger.status(),
                action_schema=action_schema,
            )
            if system_context:
                system_prompt = f"{system_context}\n\n{system_prompt}"

            # Trim message history to prevent context explosion
            if len(messages) > config.history_window:
                messages = messages[-config.history_window :]

            # Call LLM
            step_start = time.monotonic()
            try:
                response = await self._llm_chat(
                    model=self._model,
                    messages=messages,
                    system=system_prompt,
                    temperature=config.temperature,
                    max_tokens=1024,
                    num_ctx=16384,
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
                yield {
                    "type": "error",
                    "message": f"LLM call failed: {e}",
                    "iteration": iteration,
                }
                break

            step_latency = (time.monotonic() - step_start) * 1000
            raw_content = response.get("message", {}).get("content", "")

            # Record token usage
            ledger.record(
                response.get("prompt_eval_count", 0),
                response.get("eval_count", 0),
            )

            # Parse JSON action
            action = _parse_json_action(raw_content)
            if action is None:
                parse_failures += 1
                logger.warning(
                    f"JSON parse failure {parse_failures}/{config.max_parse_failures}: "
                    f"{raw_content[:200]}"
                )

                if parse_failures >= config.max_parse_failures:
                    # Fall back to chunked processing
                    yield {
                        "type": "fallback",
                        "reason": f"JSON parse failed {parse_failures} times",
                        "iteration": iteration,
                    }
                    fallback_answer = await self._chunked_fallback(question, env)
                    final_answer = fallback_answer
                    trajectory.append(
                        {
                            "iteration": iteration,
                            "action": "fallback",
                            "reason": "parse_failure",
                            "latency_ms": step_latency,
                        }
                    )
                    break

                # Tell model to try again with valid JSON
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "ERROR: Your response was not valid JSON. "
                        "Respond with ONLY a JSON object like: "
                        '{"action": "peek", "var": "$document", "start": 0, "end": 4000}',
                    }
                )
                continue

            # Reset parse failure counter on success
            parse_failures = 0
            action_type = action.get("action", "unknown")

            # Yield action event
            yield {
                "type": "action",
                "iteration": iteration,
                "action": action_type,
                "details": {
                    k: v for k, v in action.items() if k != "action" and k != "answer"
                },
                "latency_ms": round(step_latency, 1),
                "tokens": ledger.status(),
            }

            # Execute action
            exec_start = time.monotonic()
            result = await env.execute(action)
            exec_latency = (time.monotonic() - exec_start) * 1000

            # Log trajectory step
            trajectory.append(
                {
                    "iteration": iteration,
                    "action": action_type,
                    "details": {
                        k: str(v)[:200] for k, v in action.items() if k != "action"
                    },
                    "ok": result["ok"],
                    "result_preview": str(result["result"])[:300],
                    "llm_latency_ms": round(step_latency, 1),
                    "exec_latency_ms": round(exec_latency, 1),
                    "tokens_used": ledger.total,
                }
            )

            # Check for final action
            if action_type == "final":
                final_answer = result["result"]
                break

            # Yield result event
            result_preview = str(result["result"])
            if len(result_preview) > 500:
                result_preview = result_preview[:500] + "..."
            yield {
                "type": "result",
                "iteration": iteration,
                "action": action_type,
                "ok": result["ok"],
                "preview": result_preview,
                "exec_latency_ms": round(exec_latency, 1),
            }

            # Add to message history
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": str(result["result"])})

        # If we exhausted iterations without a final answer
        if final_answer is None:
            yield {
                "type": "fallback",
                "reason": "Max iterations reached without final answer",
                "iteration": config.max_iterations,
            }
            final_answer = await self._chunked_fallback(question, env)

        total_latency = (time.monotonic() - session_start) * 1000

        # Yield done event
        yield {
            "type": "done",
            "answer": final_answer,
            "session_id": session_id,
            "iterations": len(trajectory),
            "tokens_used": ledger.total,
            "prompt_tokens": ledger.prompt_tokens,
            "completion_tokens": ledger.completion_tokens,
            "latency_ms": round(total_latency, 1),
        }

        # Save trajectory (fire-and-forget)
        self._save_trajectory(
            session_id=session_id,
            question=question,
            variables=[h["name"] for h in env.handles()],
            trajectory=trajectory,
            final_answer=final_answer,
            tokens_used=ledger.total,
            latency_ms=round(total_latency, 1),
            success=final_answer is not None,
        )

    async def _chunked_fallback(self, question: str, env: REPLEnvironment) -> str:
        """
        Dumb but reliable fallback: split largest variable into chunks,
        summarize each, concatenate summaries, answer from summaries.
        Map-reduce — no JSON parsing needed.
        """
        logger.info("Recursive engine falling back to chunked processing")

        # Find the largest variable
        largest = None
        for var in env._vars.values():
            if largest is None or var.size > largest.size:
                largest = var

        if not largest:
            return "No content available to analyze."

        # Split into 6000-char chunks
        chunk_size = 6000
        chunks = []
        for i in range(0, largest.size, chunk_size):
            chunks.append(largest.content[i : i + chunk_size])

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                response = await self._llm_chat(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Summarize this text section ({i + 1}/{len(chunks)}) "
                            f"focusing on information relevant to: {question}\n\n{chunk}",
                        }
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    num_ctx=8192,
                )
                summary = response.get("message", {}).get("content", "")
                if summary:
                    summaries.append(summary)
            except Exception as e:
                logger.warning(f"Chunk {i + 1} summarization failed: {e}")
                continue

        if not summaries:
            return "Unable to process the document. Please try with a shorter text."

        # Synthesize final answer from summaries
        combined = "\n\n".join(
            f"[Section {i + 1}] {s}" for i, s in enumerate(summaries)
        )

        try:
            response = await self._llm_chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on these section summaries, answer: {question}\n\n{combined}",
                    }
                ],
                temperature=0.2,
                max_tokens=2048,
                num_ctx=16384,
            )
            return response.get("message", {}).get(
                "content", "Unable to synthesize answer."
            )
        except Exception as e:
            logger.error(f"Chunked fallback synthesis failed: {e}")
            return f"Analysis incomplete. Summaries:\n{combined}"

    def _save_trajectory(
        self,
        session_id: str,
        question: str,
        variables: List[str],
        trajectory: List[Dict[str, Any]],
        final_answer: Optional[str],
        tokens_used: int,
        latency_ms: float,
        success: bool,
    ):
        """Save trajectory to disk for Phase 5 fine-tuning."""
        try:
            traj_dir = Path(self._data_dir) / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)

            traj_file = traj_dir / f"{session_id}.json"
            data = {
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": self._model,
                "question": question,
                "variables": variables,
                "steps": trajectory,
                "final_answer": final_answer,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "iterations": len(trajectory),
                "success": success,
            }
            traj_file.write_text(json.dumps(data, indent=2))
            logger.info(
                f"Trajectory saved: {traj_file.name} "
                f"({len(trajectory)} steps, {tokens_used} tokens)"
            )
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

        # Distill trajectory into a memory
        if success and final_answer and self._memory_callback:
            answer_preview = final_answer[:200]
            var_str = ", ".join(variables[:5])
            fact = (
                f"Recursive analysis of '{question[:100]}': "
                f"{len(trajectory)} iterations, explored [{var_str}]. "
                f"Key finding: {answer_preview}"
            )
            try:
                import asyncio

                asyncio.create_task(
                    self._memory_callback(
                        fact=fact,
                        category="lessons",
                        source="recursive-trajectory",
                        metadata={
                            "session_id": session_id,
                            "iterations": len(trajectory),
                            "tokens_used": tokens_used,
                        },
                    )
                )
                logger.info(f"Trajectory memory stored for session {session_id}")
            except Exception as e:
                logger.debug(f"Trajectory memory failed: {e}")
