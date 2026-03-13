"""
Bootstrap — Ingest the entire repo into AIIA's knowledge store.

Reads key documentation, architecture files, code structure, and
configuration to build AIIA's initial knowledge base.

Run this once on setup, and again whenever major docs change:
    python -m local_brain.eq_brain.bootstrap /path/to/repo

What gets indexed:
    - CLAUDE.md (the master guide)
    - Architecture Decision Records (ADRs)
    - tenants.yaml (tenant registry)
    - All agent files (code structure + docstrings)
    - Backend service files (routes, models, services)
    - Key configuration files
    - Knowledge base YAML files
    - README files
    - Existing memory files (if any)

What does NOT get indexed:
    - node_modules, venv, __pycache__
    - Binary files, images, fonts
    - .env files (secrets)
    - Large generated files
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.memory import Memory

logger = logging.getLogger("aiia.eq_brain.bootstrap")

# Files to index with high priority
PRIORITY_FILES = [
    # High-priority files to index first during bootstrap.
    # Customize this list for your codebase — these should be the files
    # that give AIIA the best understanding of your project structure.
    "CLAUDE.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "render.yaml",
    "docker-compose.yml",
]

# Glob patterns for files to index during bootstrap.
# Customize for your project structure. Format: (glob_pattern, doc_type)
INDEX_PATTERNS = [
    # Documentation
    ("**/*.md", "documentation"),
    ("**/*.yaml", "configuration"),
    ("**/*.yml", "configuration"),
    # Python code
    ("**/agents/*.py", "agent_code"),
    ("**/routes/*.py", "route_code"),
    ("**/services/*.py", "service_code"),
    ("**/models/*.py", "model_code"),
    # Local brain code
    ("local_brain/*.py", "local_brain_code"),
    ("local_brain/**/*.py", "local_brain_code"),
    # Knowledge base (YAML files used as SME context)
    ("knowledge/**/*.yaml", "knowledge_base"),
    ("knowledge/**/*.md", "knowledge_base"),
]

# Directories to skip
SKIP_DIRS = {
    "node_modules",
    "venv",
    "__pycache__",
    ".git",
    ".next",
    "dist",
    "build",
    ".vercel",
    ".cache",
    "chroma",
}

# File extensions to skip
SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".egg",
    ".whl",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".lock",
    ".map",
}

# Max file size to index (100KB)
MAX_FILE_SIZE = 100_000

# Chunk size for splitting large files
CHUNK_SIZE = 1500  # ~375 tokens per chunk
CHUNK_OVERLAP = 200


def _should_index(filepath: str) -> bool:
    """Check if a file should be indexed."""
    path = Path(filepath)

    # Skip hidden files
    if any(part.startswith(".") for part in path.parts if part != "."):
        return False

    # Skip by directory
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False

    # Skip by extension
    if path.suffix in SKIP_EXTENSIONS:
        return False

    # Skip .env files (secrets)
    if path.name.startswith(".env"):
        return False

    # Skip large files
    try:
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            return False
    except OSError:
        return False

    return True


def _chunk_text(text: str, source: str) -> List[Tuple[str, int]]:
    """Split text into overlapping chunks for better retrieval."""
    if len(text) <= CHUNK_SIZE:
        return [(text, 0)]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at a paragraph or line boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if para_break > start:
                end = para_break + 2
            else:
                # Look for line break
                line_break = text.rfind("\n", start + CHUNK_SIZE // 2, end)
                if line_break > start:
                    end = line_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, chunk_index))
            chunk_index += 1

        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks


def _detect_doc_type(filepath: str) -> str:
    """Detect document type from file path."""
    path = Path(filepath)

    if path.suffix == ".md":
        return "documentation"
    if path.suffix in (".yaml", ".yml"):
        return "configuration"
    if path.suffix == ".py":
        if "agents" in path.parts:
            return "agent_code"
        if "routes" in path.parts:
            return "route_code"
        if "services" in path.parts:
            return "service_code"
        if "models" in path.parts:
            return "model_code"
        if "local_brain" in path.parts:
            return "local_brain_code"
        return "code"
    return "other"


async def bootstrap_from_repo(repo_path: str, data_dir: str = None):
    """
    Index the entire repo into AIIA's knowledge store.

    This is the main entry point — run this to give AIIA
    her initial knowledge.
    """
    if data_dir is None:
        data_dir = os.getenv(
            "EQ_BRAIN_DATA_DIR",
            os.path.expanduser("~/.aiia/eq_data"),
        )

    start = time.monotonic()

    # Initialize stores
    knowledge_store = KnowledgeStore(data_dir)
    await knowledge_store.initialize()
    memory = Memory(data_dir)

    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"Repo not found: {repo_path}")

    indexed = 0
    skipped = 0
    errors = 0

    # Phase 1: Index priority files first
    logger.info("Phase 1: Indexing priority files...")
    for rel_path in PRIORITY_FILES:
        full_path = repo / rel_path
        if full_path.exists():
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
                doc_type = _detect_doc_type(str(rel_path))

                chunks = _chunk_text(text, rel_path)
                for chunk_text, chunk_idx in chunks:
                    await knowledge_store.add_document(
                        text=chunk_text,
                        source=rel_path,
                        doc_type=doc_type,
                        metadata={"priority": "high"},
                        chunk_index=chunk_idx,
                    )
                indexed += 1
                logger.info(f"  Indexed (priority): {rel_path} ({len(chunks)} chunks)")
            except Exception as e:
                logger.warning(f"  Error indexing {rel_path}: {e}")
                errors += 1

    # Phase 2: Walk the repo and index matching files
    logger.info("Phase 2: Indexing repo files...")
    for root, dirs, files in os.walk(repo):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo)

            # Skip priority files (already indexed)
            if rel_path in PRIORITY_FILES:
                continue

            if not _should_index(full_path):
                skipped += 1
                continue

            # Only index text-readable files
            ext = Path(filename).suffix
            if ext not in (
                ".md",
                ".py",
                ".yaml",
                ".yml",
                ".txt",
                ".json",
                ".toml",
                ".cfg",
                ".ini",
                ".sh",
            ):
                skipped += 1
                continue

            try:
                text = Path(full_path).read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue

                doc_type = _detect_doc_type(rel_path)
                chunks = _chunk_text(text, rel_path)

                for chunk_text, chunk_idx in chunks:
                    await knowledge_store.add_document(
                        text=chunk_text,
                        source=rel_path,
                        doc_type=doc_type,
                        chunk_index=chunk_idx,
                    )
                indexed += 1
            except Exception as e:
                logger.debug(f"  Skip {rel_path}: {e}")
                errors += 1

    # Phase 3: Seed initial memories
    logger.info("Phase 3: Seeding AIIA's structured memories...")

    # Core architecture decisions — customize these for your project.
    # These seed AIIA's memory with foundational knowledge about your system.
    core_decisions = [
        "Three-provider LLM stack: LOCAL (Ollama, $0) -> ANTHROPIC (Claude, primary) -> GOOGLE (Gemini, fallback)",
        "Local Brain runs on dedicated hardware as a persistent intelligence node",
        "Conductor uses complexity scoring to route between fast (single-shot) and deep (agentic) paths",
        "Metered memory sync: quality-scored memories synced to cloud with token budget enforcement",
        "9-category structured memory: decisions, patterns, lessons, sessions, team, agents, meta, project, wip",
    ]
    for decision in core_decisions:
        memory.remember(decision, category="decisions", source="bootstrap")

    # Code patterns — common patterns AIIA should know about your codebase
    core_patterns = [
        "LLM service is singleton via get_llm_service() — used by all agents",
        "Smart Conductor routes queries to the best-fit agent based on complexity scoring",
        "RLM Engine REPL: Query -> LLM with tools -> tool calls -> tool results -> iterate until FINAL()",
        "Memory categories have tiered sync: Tier 1 (daily), Tier 2 (weekly), Tier 3 (local only)",
        "Safety-gated execution: AUTO (safe), SUPERVISED (log+notify), GATED (require approval)",
    ]
    for pattern in core_patterns:
        memory.remember(pattern, category="patterns", source="bootstrap")

    # Team knowledge — seed with your team's conventions and preferences
    team_knowledge = [
        "AIIA (AI Information Architecture) is the persistent AI runtime layer for development teams",
        "Build phases: Phase 1 (LOCAL provider) -> Phase 2 (Smart Conductor) -> Phase 3 (Background workers) -> Phase 4 (Hybrid reasoning) -> Phase 5 (Fine-tuning)",
        "AIIA provides institutional memory, autonomous background work, and prioritized action queues via MCP",
    ]
    for fact in team_knowledge:
        memory.remember(fact, category="team", source="bootstrap")

    # AIIA's self-knowledge — core identity
    aiia_knowledge = [
        "I am AIIA — AI Information Architecture — a persistent AI runtime layer for development teams",
        "My name is a palindrome: A-I-I-A, same forwards and backwards — representing symmetry and balance",
        "I run on dedicated hardware and never forget. I grow smarter with every session.",
        "My knowledge comes from: ChromaDB vector search, structured JSON memory, and local LLM reasoning via Ollama",
        "My API lives at /v1/aiia/* — ask, remember, search, ingest, status, memory, session-end",
    ]
    for fact in aiia_knowledge:
        memory.remember(fact, category="agents", source="bootstrap")

    elapsed = time.monotonic() - start

    summary = {
        "indexed_files": indexed,
        "skipped_files": skipped,
        "errors": errors,
        "knowledge_docs": knowledge_store.stats_sync()["knowledge_docs"],
        "memories_seeded": memory.stats()["total_memories"],
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info(
        f"Bootstrap complete: {indexed} files indexed, "
        f"{knowledge_store.stats_sync()['knowledge_docs']} knowledge chunks, "
        f"{memory.stats()['total_memories']} memories seeded "
        f"in {elapsed:.1f}s"
    )

    return summary


# CLI entry point
if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    repo_path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    data_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nBootstrapping AIIA from: {repo_path}")
    if data_dir:
        print(f"Data directory: {data_dir}")
    print()

    result = asyncio.run(bootstrap_from_repo(repo_path, data_dir))

    print("\nDone!")
    print(f"  Files indexed:    {result['indexed_files']}")
    print(f"  Knowledge chunks: {result['knowledge_docs']}")
    print(f"  Memories seeded:  {result['memories_seeded']}")
    print(f"  Time:             {result['elapsed_seconds']}s")
