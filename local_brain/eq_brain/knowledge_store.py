"""
Knowledge Store — ChromaDB vector store for AIIA.

Indexes repo documents, architecture decisions, code patterns, and
session transcripts. Everything AIIA needs to answer questions
grounded in the actual codebase.

Uses ChromaDB's built-in MiniLM embeddings — no cloud dependency.
All data stays on the Mac Mini.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.eq_brain.knowledge")


class KnowledgeStore:
    """
    ChromaDB-backed vector store for the EQ Brain.

    Collections:
        - aiia_knowledge: Main knowledge base (docs, architecture, code)
        - aiia_sessions: Session summaries and transcripts
    """

    def __init__(self, data_dir: str, collection_name: str = "aiia_knowledge"):
        self._data_dir = data_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._sessions_collection = None

    async def initialize(self):
        """Initialize ChromaDB client and collections."""
        import chromadb

        chroma_dir = os.path.join(self._data_dir, "chroma")
        os.makedirs(chroma_dir, exist_ok=True)

        # Disable telemetry via env var (avoids Settings pydantic issues)
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        self._client = chromadb.PersistentClient(path=chroma_dir)

        # Main knowledge collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "AIIA knowledge base"},
        )

        # Session memory collection
        self._sessions_collection = self._client.get_or_create_collection(
            name=f"{self._collection_name}_sessions",
            metadata={"description": "Session summaries and transcripts"},
        )

        logger.info(
            f"KnowledgeStore initialized: {self._collection.count()} docs, "
            f"{self._sessions_collection.count()} sessions"
        )

    def _doc_id(self, source: str, chunk_index: int = 0) -> str:
        """Generate a deterministic document ID from source path + chunk index."""
        content = f"{source}::{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def add_document(
        self,
        text: str,
        source: str,
        doc_type: str = "documentation",
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
    ):
        """Add a document chunk to the knowledge store."""
        if not self._collection:
            raise RuntimeError("KnowledgeStore not initialized")

        doc_id = self._doc_id(source, chunk_index)

        doc_metadata = {
            "source": source,
            "doc_type": doc_type,
            "chunk_index": chunk_index,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if metadata:
            doc_metadata.update(metadata)

        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[doc_metadata],
        )

    async def add_documents(
        self,
        texts: List[str],
        sources: List[str],
        doc_type: str = "documentation",
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add multiple document chunks at once."""
        if not self._collection:
            raise RuntimeError("KnowledgeStore not initialized")

        ids = [self._doc_id(src, i) for i, src in enumerate(sources)]

        base_metadatas = []
        for i, src in enumerate(sources):
            m = {
                "source": src,
                "doc_type": doc_type,
                "chunk_index": i,
                "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if metadatas and i < len(metadatas):
                m.update(metadatas[i])
            base_metadatas.append(m)

        # ChromaDB has a batch limit, chunk to 100 at a time
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=base_metadatas[start:end],
            )

    async def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge store for relevant documents."""
        if not self._collection:
            raise RuntimeError("KnowledgeStore not initialized")

        where_filter = None
        if doc_type:
            where_filter = {"doc_type": doc_type}

        results = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        documents = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                documents.append(
                    {
                        "content": doc,
                        "source": meta.get("source", "unknown"),
                        "doc_type": meta.get("doc_type", "unknown"),
                        "relevance": round(
                            1 - distance, 3
                        ),  # Convert distance to relevance
                        "metadata": meta,
                    }
                )

        return documents

    async def add_session(
        self,
        summary: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store a session summary for long-term recall."""
        if not self._sessions_collection:
            raise RuntimeError("KnowledgeStore not initialized")

        doc_metadata = {
            "session_id": session_id,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if metadata:
            doc_metadata.update(metadata)

        self._sessions_collection.upsert(
            ids=[session_id],
            documents=[summary],
            metadatas=[doc_metadata],
        )

    async def search_sessions(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search past session summaries."""
        if not self._sessions_collection:
            return []

        results = await asyncio.to_thread(
            self._sessions_collection.query,
            query_texts=[query],
            n_results=n_results,
        )

        sessions = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                sessions.append(
                    {
                        "summary": doc,
                        "session_id": meta.get("session_id", "unknown"),
                        "metadata": meta,
                    }
                )

        return sessions

    async def stats(self) -> Dict[str, Any]:
        """Return knowledge store statistics (async, non-blocking)."""
        k_count = await asyncio.to_thread(self._collection.count) if self._collection else 0
        s_count = await asyncio.to_thread(self._sessions_collection.count) if self._sessions_collection else 0
        return {
            "knowledge_docs": k_count,
            "session_docs": s_count,
            "data_dir": self._data_dir,
            "collection": self._collection_name,
        }

    def stats_sync(self) -> Dict[str, Any]:
        """Return knowledge store statistics (sync, for scripts)."""
        return {
            "knowledge_docs": self._collection.count() if self._collection else 0,
            "session_docs": self._sessions_collection.count() if self._sessions_collection else 0,
            "data_dir": self._data_dir,
            "collection": self._collection_name,
        }
