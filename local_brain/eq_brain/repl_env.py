"""
REPL Environment for Recursive Inference.

Stores variables (documents, knowledge, memory) as named handles.
The LLM never sees full content in context — only name, type, size,
and a short preview. Actions (peek, search, store, etc.) let the
model explore variables incrementally within its context budget.

This is the variable storage + action execution layer. The loop
itself lives in recursive_engine.py.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("aiia.eq_brain.repl_env")

# Limits
PEEK_MAX_CHARS = 8000
SEARCH_WINDOW_CHARS = 1500
SEARCH_MAX_RESULTS = 5
PREVIEW_CHARS = 200
SUB_ASK_CONTEXT_CAP = 6000
CHUNK_SUMMARIZE_MAX = 8000

# LLM callback type: async (model, messages, system, temperature, max_tokens) -> str
LLMCallback = Callable[..., Coroutine[Any, Any, str]]


@dataclass
class REPLVariable:
    """A named variable in the REPL environment."""

    name: str
    content: str
    var_type: str  # "document", "knowledge", "memory", "sessions", "derived"
    readonly: bool = True

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def preview(self) -> str:
        return self.content[:PREVIEW_CHARS].replace("\n", " ")

    def handle(self) -> Dict[str, Any]:
        """Compact descriptor the model sees instead of full content."""
        return {
            "name": self.name,
            "type": self.var_type,
            "size": self.size,
            "preview": self.preview,
        }


class REPLEnvironment:
    """
    Variable store + action executor for recursive inference.

    Load variables at the start of a session, then call execute()
    with action dicts from the model. Each action returns a result
    string that gets fed back as the next user message.
    """

    def __init__(self, llm_callback: Optional[LLMCallback] = None, max_depth: int = 3):
        self._vars: Dict[str, REPLVariable] = {}
        self._llm_callback = llm_callback
        self._max_depth = max_depth
        self._current_depth = 0

    def load(self, name: str, content: str, var_type: str, readonly: bool = True):
        """Load a variable into the environment."""
        if not content:
            return
        self._vars[name] = REPLVariable(
            name=name, content=content, var_type=var_type, readonly=readonly
        )
        logger.debug(f"REPL loaded ${name} ({var_type}, {len(content)} chars)")

    def handles(self) -> List[Dict[str, Any]]:
        """Return compact handles for all variables (what the model sees)."""
        return [v.handle() for v in self._vars.values()]

    async def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a model-requested action.

        Args:
            action: Dict with "action" key + action-specific fields.

        Returns:
            {"ok": True/False, "result": str, "action": str}
        """
        action_type = action.get("action", "").lower()

        dispatch = {
            "peek": self._action_peek,
            "search": self._action_search,
            "chunk_summarize": self._action_chunk_summarize,
            "store": self._action_store,
            "sub_ask": self._action_sub_ask,
            "final": self._action_final,
        }

        handler = dispatch.get(action_type)
        if not handler:
            return {
                "ok": False,
                "action": action_type,
                "result": f"Unknown action: {action_type}. "
                f"Valid: {', '.join(dispatch.keys())}",
            }

        try:
            result = await handler(action)
            return {"ok": True, "action": action_type, "result": result}
        except Exception as e:
            logger.warning(f"REPL action {action_type} failed: {e}")
            return {"ok": False, "action": action_type, "result": str(e)}

    def _get_var(self, action: Dict[str, Any], key: str = "var") -> REPLVariable:
        """Get a variable by name from the action dict. Raises KeyError."""
        name = action.get(key, "")
        if name.startswith("$"):
            name = name[1:]
        if name not in self._vars:
            raise KeyError(
                f"Variable ${name} not found. "
                f"Available: {', '.join('$' + n for n in self._vars)}"
            )
        return self._vars[name]

    # ── Actions ──────────────────────────────────────────────────

    async def _action_peek(self, action: Dict[str, Any]) -> str:
        """Return a character slice of a variable."""
        var = self._get_var(action)
        start = max(0, int(action.get("start", 0)))
        end = int(action.get("end", start + PEEK_MAX_CHARS))
        end = min(end, start + PEEK_MAX_CHARS, var.size)

        chunk = var.content[start:end]
        return f"[${var.name} chars {start}-{end} of {var.size}]\n{chunk}"

    async def _action_search(self, action: Dict[str, Any]) -> str:
        """Keyword search within a variable. Returns up to 5 matches with context windows."""
        var = self._get_var(action)
        query = action.get("query", "")
        if not query:
            return "Error: 'query' is required for search action."

        content_lower = var.content.lower()
        query_lower = query.lower()

        # Split query into terms for broader matching
        terms = query_lower.split()
        matches: List[Dict[str, Any]] = []
        seen_positions: set = set()

        for term in terms:
            pos = 0
            while len(matches) < SEARCH_MAX_RESULTS:
                idx = content_lower.find(term, pos)
                if idx == -1:
                    break
                # Dedup overlapping windows
                bucket = idx // (SEARCH_WINDOW_CHARS // 2)
                if bucket in seen_positions:
                    pos = idx + len(term)
                    continue
                seen_positions.add(bucket)

                window_start = max(0, idx - SEARCH_WINDOW_CHARS // 3)
                window_end = min(var.size, idx + SEARCH_WINDOW_CHARS * 2 // 3)
                window = var.content[window_start:window_end]
                matches.append(
                    {
                        "position": idx,
                        "window_start": window_start,
                        "window_end": window_end,
                        "text": window,
                    }
                )
                pos = idx + len(term)

        if not matches:
            return f"No matches for '{query}' in ${var.name} ({var.size} chars)."

        parts = [f"Found {len(matches)} match(es) for '{query}' in ${var.name}:"]
        for i, m in enumerate(matches):
            parts.append(
                f"\n--- Match {i + 1} (position {m['position']}, "
                f"chars {m['window_start']}-{m['window_end']}) ---\n{m['text']}"
            )
        return "\n".join(parts)

    async def _action_chunk_summarize(self, action: Dict[str, Any]) -> str:
        """Summarize a slice of a variable using the LLM."""
        if not self._llm_callback:
            return "Error: LLM callback not configured for chunk_summarize."

        var = self._get_var(action)
        start = max(0, int(action.get("start", 0)))
        end = int(action.get("end", start + CHUNK_SUMMARIZE_MAX))
        end = min(end, start + CHUNK_SUMMARIZE_MAX, var.size)

        chunk = var.content[start:end]
        summary = await self._llm_callback(
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this text concisely:\n\n{chunk}",
                }
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return f"[Summary of ${var.name} chars {start}-{end}]\n{summary}"

    async def _action_store(self, action: Dict[str, Any]) -> str:
        """Store a derived value as a new variable."""
        name = action.get("name", "")
        value = action.get("value", "")
        if not name:
            return "Error: 'name' is required for store action."
        if not value:
            return "Error: 'value' is required for store action."

        # Clean leading $
        if name.startswith("$"):
            name = name[1:]

        # Can't overwrite readonly vars
        if name in self._vars and self._vars[name].readonly:
            return f"Error: ${name} is readonly. Use a different name."

        self._vars[name] = REPLVariable(
            name=name, content=value, var_type="derived", readonly=False
        )
        return f"Stored ${name} ({len(value)} chars)."

    async def _action_sub_ask(self, action: Dict[str, Any]) -> str:
        """Ask a sub-question using a variable as context. Depth-limited."""
        if not self._llm_callback:
            return "Error: LLM callback not configured for sub_ask."

        if self._current_depth >= self._max_depth:
            return (
                f"Error: Max recursion depth ({self._max_depth}) reached. "
                "Use peek/search to gather info, then produce a final answer."
            )

        question = action.get("question", "")
        if not question:
            return "Error: 'question' is required for sub_ask."

        # Get context variable
        context_var_name = action.get("context_var", "")
        context = ""
        if context_var_name:
            try:
                var = self._get_var(action, key="context_var")
                context = var.content[:SUB_ASK_CONTEXT_CAP]
            except KeyError as e:
                return str(e)

        self._current_depth += 1
        try:
            messages = [{"role": "user", "content": question}]
            system = f"Answer based on this context:\n\n{context}" if context else None
            answer = await self._llm_callback(
                messages=messages,
                system=system,
                temperature=0.2,
                max_tokens=1024,
            )
            return f"[Sub-answer (depth {self._current_depth})]\n{answer}"
        finally:
            self._current_depth -= 1

    async def _action_final(self, action: Dict[str, Any]) -> str:
        """Model's final synthesized answer. Terminates the REPL loop."""
        answer = action.get("answer", "")
        if not answer:
            return "Error: 'answer' is required for final action."
        return answer

    # ── Action schema for system prompt ──────────────────────────

    @staticmethod
    def action_schema() -> str:
        """Return the action schema description for the system prompt."""
        return """Available actions (respond with ONLY valid JSON):

1. peek — Read a slice of a variable
   {"action": "peek", "var": "$document", "start": 0, "end": 4000}

2. search — Find keywords in a variable
   {"action": "search", "var": "$document", "query": "key finding"}

3. chunk_summarize — Summarize a slice (uses LLM sub-call)
   {"action": "chunk_summarize", "var": "$document", "start": 0, "end": 6000}

4. store — Save intermediate results for later use
   {"action": "store", "name": "findings", "value": "The key points are..."}

5. sub_ask — Ask a focused question with a variable as context
   {"action": "sub_ask", "question": "What is the main argument?", "context_var": "$document"}

6. final — Return your synthesized answer (terminates the session)
   {"action": "final", "answer": "Based on my analysis..."}

Strategy:
- Start with peek to scan the beginning and end of the document
- Use search to find sections relevant to the question
- Use peek to read those sections in detail
- Use store to save intermediate findings
- Use final when you have enough information to answer"""
