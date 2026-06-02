"""
AIIA Research Harness — autonomous deep research with RAG and persistent memory.

A topic is a named research question. Each run fetches sources, indexes them
into ChromaDB, tracks open gaps, and updates a running synthesis. Sessions
accumulate — the corpus and synthesis grow across runs until the question is answered.
"""
