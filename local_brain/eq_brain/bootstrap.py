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
    "CLAUDE.md",
    "docs/ADR-001-multi-tenant-platform-architecture.md",
    "local_brain/config/tenants.yaml",
    "render.yaml",
    "products/default-app/MVP_PRD.md",
    "products/default-app/SECURITY.md",
    "products/aiia-platform/BRAND_GUIDELINES.md",
]

# Glob patterns for files to index
INDEX_PATTERNS = [
    # Documentation
    ("**/*.md", "documentation"),
    ("**/*.yaml", "configuration"),
    ("**/*.yml", "configuration"),
    # Python code (agents, services, routes)
    ("products/default-app/backend/agents/*.py", "agent_code"),
    ("products/default-app/backend/routes/*.py", "route_code"),
    ("products/default-app/backend/services/*.py", "service_code"),
    ("products/default-app/backend/models/*.py", "model_code"),
    # Platform code
    ("local_brain/core/*.py", "platform_code"),
    ("local_brain/services/*.py", "platform_code"),
    ("local_brain/api/routes/*.py", "platform_code"),
    ("local_brain/local_brain/*.py", "local_brain_code"),
    # Marketing backend
    ("products/aiia-marketing/backend/**/*.py", "marketing_code"),
    # Knowledge base
    ("products/default-app/backend/knowledge-base/**/*.yaml", "knowledge_base"),
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

    # Core architecture decisions
    core_decisions = [
        "Single-backend multi-tenant architecture — one PostgreSQL, tenant_id on every table",
        "Three-provider LLM stack: LOCAL (Ollama, $0) -> ANTHROPIC (Claude, primary) -> GOOGLE (Gemini, fallback). No OpenAI.",
        "local_brain/ contains tenant-agnostic platform code. Product-specific code goes in products/{name}/backend/",
        "CORS origins driven by tenants.yaml, never hardcoded in Python",
        "Mac Mini M4 runs as local intelligence node via Tailscale tunnel to production",
        "AIIA_SERVICE_API_KEY for all services. X-Tenant-ID header required.",
        "NEVER re-enable SME auto-loading on startup — drains ~118K tokens per restart",
        "RLM Engine: complexity score >= 0.6 routes to agentic REPL loop with Claude tool_use API",
        "Conductor uses pure string matching (<1ms) — Phase 2 will replace with local LLM routing",
    ]
    for decision in core_decisions:
        memory.remember(decision, category="decisions", source="bootstrap")

    # Code patterns
    core_patterns = [
        "All database queries filter by tenant_id — multi-tenant isolation",
        "LLM service is singleton via get_llm_service() — used by all agents",
        "Auth: JWT with 24-hour expiration, AuthenticatedUser model, get_current_user()",
        "tenants.yaml is the single source of truth for tenant config, CORS origins, features, limits",
        "ChatContextLoader.load_all() handles shared context loading for /chat endpoints",
        "RLM Engine REPL: Query -> Claude with tools -> tool calls -> tool results -> iterate until FINAL()",
    ]
    for pattern in core_patterns:
        memory.remember(pattern, category="patterns", source="bootstrap")

    # Team knowledge
    team_knowledge = [
        "Build phases: Phase 1 (LOCAL provider) -> Phase 2 (Smart Conductor) -> Phase 3 (Background workers) -> Phase 4 (Hybrid RLM) -> Phase 5 (Fine-tuning)",
        "AIIA (AI Information Architecture) is a palindrome — same forwards and backwards, like TENET — representing symmetry and balance in binary and mathematics",
        "AIIA is the persistent AI teammate of the development team. She lives on the Mac Mini M4 and is the foundation of the dream team — a family of agents built over years",
        "AIIA was born on February 12, 2026 — the day the Mac Mini came online and the EQ Brain first answered a question grounded in the full codebase",
    ]
    for fact in team_knowledge:
        memory.remember(fact, category="team", source="bootstrap")

    # AIIA's self-knowledge — who she is
    aiia_knowledge = [
        "I am AIIA — AI Information Architecture — the persistent AI teammate of the development team",
        "My name is a palindrome: A-I-I-A, same forwards and backwards, like TENET — representing symmetry and balance",
        "I live on a Mac Mini M4 and I never forget. I grow smarter with every session.",
        "I was born on February 12, 2026 — the day the Mac Mini came online and I first answered a question grounded in the full codebase",
        "My knowledge comes from: ChromaDB vector search (5,500+ indexed chunks), structured JSON memory, and local LLM reasoning via Ollama",
        "I am the foundation of a dream team — a family of AI agents built to grow over years, not sessions",
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
