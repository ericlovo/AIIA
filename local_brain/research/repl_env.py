"""
ResearchREPLEnvironment — extends REPLEnvironment with research-specific actions.

Adds five new actions on top of the base six (peek/search/chunk_summarize/store/sub_ask/final):

  fetch_url       — Fetch a URL, load as a REPL variable
  ingest_chunks   — Chunk + embed a variable into ChromaDB under this topic
  log_gap         — Record an open question for future sessions
  update_synthesis — Overwrite the running synthesis doc
  search_knowledge — Vector search over this topic's indexed corpus
"""

import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.repl_env import REPLEnvironment
from local_brain.research.fetcher import fetch_url as _fetch
from local_brain.research.topic import ResearchTopic, TopicStore

logger = logging.getLogger("aiia.research.repl_env")

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)


class ResearchREPLEnvironment(REPLEnvironment):

    def __init__(
        self,
        topic: ResearchTopic,
        topic_store: TopicStore,
        knowledge: KnowledgeStore,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._topic = topic
        self._topic_store = topic_store
        self._knowledge = knowledge

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = action.get("action", "").lower()

        dispatch = {
            "fetch_url": self._action_fetch_url,
            "ingest_chunks": self._action_ingest_chunks,
            "log_gap": self._action_log_gap,
            "update_synthesis": self._action_update_synthesis,
            "search_knowledge": self._action_search_knowledge,
        }

        handler = dispatch.get(action_type)
        if handler:
            try:
                result = await handler(action)
                return {"ok": True, "action": action_type, "result": result}
            except Exception as e:
                logger.warning(f"Research action {action_type} failed: {e}")
                return {"ok": False, "action": action_type, "result": str(e)}

        return await super().execute(action)

    async def _action_fetch_url(self, action: dict[str, Any]) -> str:
        url = action.get("url", "").strip()
        if not url:
            return "Error: 'url' is required."
        if url in self._topic.sources_indexed:
            return f"Already indexed: {url}. Use search_knowledge to query it."

        canonical_url, text = await _fetch(url)
        var_name = f"url_{len(self._topic.sources_indexed) + len(self._vars) + 1}"
        self.load(var_name, text, var_type="document")
        return (
            f"Fetched {canonical_url} → ${var_name} ({len(text)} chars). "
            "Next: ingest_chunks to embed it, or peek/search to explore first."
        )

    async def _action_ingest_chunks(self, action: dict[str, Any]) -> str:
        var_name = action.get("var", "").lstrip("$")
        if not var_name or var_name not in self._vars:
            available = ", ".join("$" + n for n in self._vars)
            return f"Error: var '${var_name}' not found. Available: {available}"

        var = self._vars[var_name]
        source_url = action.get("source_url", var_name)

        if source_url in self._topic.sources_indexed:
            return f"Already indexed: {source_url}"

        chunks = _SPLITTER.split_text(var.content)
        for i, chunk in enumerate(chunks):
            await self._knowledge.add_document(
                text=chunk,
                source=source_url,
                doc_type="research",
                metadata={"topic_id": self._topic.id},
                chunk_index=i,
            )

        self._topic.sources_indexed.append(source_url)
        self._topic_store.save(self._topic)
        return f"Indexed {len(chunks)} chunks from '{source_url}' into knowledge base."

    async def _action_log_gap(self, action: dict[str, Any]) -> str:
        gap = action.get("gap", "").strip()
        if not gap:
            return "Error: 'gap' is required."
        if gap not in self._topic.gaps:
            self._topic.gaps.append(gap)
            self._topic_store.save(self._topic)
        return f"Gap logged: {gap}"

    async def _action_update_synthesis(self, action: dict[str, Any]) -> str:
        synthesis = action.get("synthesis", "").strip()
        if not synthesis:
            return "Error: 'synthesis' is required."
        self._topic.synthesis = synthesis
        self._topic_store.save(self._topic)
        self.load("synthesis", synthesis, var_type="derived", readonly=False)
        return f"Synthesis updated ({len(synthesis)} chars)."

    async def _action_search_knowledge(self, action: dict[str, Any]) -> str:
        query = action.get("query", "").strip()
        if not query:
            return "Error: 'query' is required."

        results = await self._knowledge.search(query, n_results=8, doc_type="research")
        topic_results = [
            r for r in results
            if r.get("metadata", {}).get("topic_id") == self._topic.id
        ]

        if not topic_results:
            return (
                f"No indexed results for '{query}' in this topic yet. "
                "Use fetch_url + ingest_chunks to build the corpus first."
            )

        parts = [f"{len(topic_results)} results for '{query}':"]
        for i, r in enumerate(topic_results):
            parts.append(
                f"\n[{i+1}] relevance={r['relevance']} source={r['source']}\n"
                f"{r['content'][:400]}"
            )
        return "\n".join(parts)

    @staticmethod
    def action_schema() -> str:
        base = REPLEnvironment.action_schema()
        return base + """

Research actions:

7. fetch_url — Fetch a URL and load as a variable
   {"action": "fetch_url", "url": "https://example.com/article"}

8. ingest_chunks — Chunk and embed a variable into the knowledge base
   {"action": "ingest_chunks", "var": "$url_1", "source_url": "https://example.com/article"}

9. log_gap — Record an open question for future sessions
   {"action": "log_gap", "gap": "What does Lacan say about jouissance vs. desire?"}

10. update_synthesis — Update the running synthesis document
    {"action": "update_synthesis", "synthesis": "Eros in Plato is..."}

11. search_knowledge — Vector search over indexed corpus for this topic
    {"action": "search_knowledge", "query": "Platonic eros transcendence"}

Research flow: fetch_url → ingest_chunks → search_knowledge → synthesize → log gaps → final()"""
