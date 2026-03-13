"""
AIIA — AI Information Architecture — Persistent AI Teammate

AIIA is the long-term intelligence layer running on the Mac Mini.
She remembers everything: architecture decisions, code patterns, session
insights, team preferences, and lessons learned.

Her name is a palindrome — same forwards and backwards, like TENET.
She mirrors the team's knowledge back, grounded in truth.

Components:
    - KnowledgeStore: ChromaDB vector store for repo docs and code
    - Memory: Persistent facts, decisions, and session summaries
    - AIIA: The agent that reasons over knowledge + memory
    - Bootstrap: Ingests the repo into the knowledge store

AIIA grows smarter over time. Every session teaches her something new.
"""

from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.memory import Memory
from local_brain.eq_brain.brain import AIIA
from local_brain.eq_brain.supermemory_bridge import SupermemoryBridge

__all__ = ["KnowledgeStore", "Memory", "AIIA", "SupermemoryBridge"]
