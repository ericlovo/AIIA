"""
Session Indexer Runner — CLI entry point for Claude Code transcript indexing.

Parses JSONL session transcripts from ~/.claude/projects/, extracts structured
knowledge via heuristics + local LLM ($0), stores in ChromaDB + AIIA memory.

Usage:
    python -m local_brain.scripts.session_indexer_runner [--force]
    # Or from brain CLI: brain session-index [--force]

Schedule:
    Daily at 5:30am via com.aiia.sessionindex launchd agent
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging before imports that use it
LOG_DIR = Path.home() / ".aiia" / "logs" / "session-index"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "session-index.log"),
    ],
)
logger = logging.getLogger("aiia.session_indexer_runner")


def main():
    # Parse args
    force = "--force" in sys.argv

    logger.info(f"Session indexer starting — force={force}")

    # Load .env if running standalone
    env_path = Path.home() / ".aiia" / ".env"
    if env_path.exists():
        logger.info("Loading .env")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and not os.getenv(key):
                        os.environ[key] = value

    # Set PYTHONPATH so imports work
    repo_dir = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(repo_dir))

    try:
        result = asyncio.run(_run_indexer(force))
        _write_report(result)
        return 0 if not result.get("errors") else 1
    except Exception as e:
        logger.error(f"Session indexer failed: {e}", exc_info=True)
        return 1


async def _run_indexer(force: bool) -> dict:
    """Run the session indexing pipeline."""
    from local_brain.config import get_config
    from local_brain.eq_brain.knowledge_store import KnowledgeStore
    from local_brain.eq_brain.memory import Memory
    from local_brain.eq_brain.session_indexer import SessionIndexer
    from local_brain.eq_brain.supermemory_bridge import (
        SupermemoryBridge,
    )
    from local_brain.ollama_client import OllamaClient

    config = get_config()

    # Check Ollama health first
    ollama = OllamaClient(config=config)
    health = await ollama.health()

    ollama_available = health["status"] == "online"
    if not ollama_available:
        logger.warning(
            f"Ollama offline: {health.get('error', 'unknown')} — "
            "will index with heuristics only (no LLM enrichment)"
        )

    # Initialize stores
    memory = Memory(data_dir=config.eq_brain_data_dir)
    knowledge = KnowledgeStore(
        data_dir=config.eq_brain_data_dir,
        collection_name=config.eq_brain_collection,
    )
    await knowledge.initialize()

    # Determine enrichment model
    task_model = config.models.get("task")
    enrichment_model = task_model.model_name if task_model else "qwen2.5:7b"

    # Initialize Supermemory bridge for cloud sync
    bridge = SupermemoryBridge(timeout=config.supermemory_timeout)
    bridge_available = bridge.available
    if bridge_available:
        logger.info("Supermemory bridge available — sessions will sync to cloud")
    else:
        logger.info("Supermemory bridge unavailable — local-only indexing")

    # Create indexer
    indexer = SessionIndexer(
        knowledge_store=knowledge,
        memory=memory,
        ollama_client=ollama if ollama_available else None,
        data_dir=config.eq_brain_data_dir,
        enrichment_model=enrichment_model,
        supermemory_bridge=bridge if bridge_available else None,
    )

    # Log pre-run state
    mem_stats = memory.stats()
    kb_stats = knowledge.stats_sync_sync()
    logger.info(
        f"Memory: {mem_stats['total_memories']} total, "
        f"ChromaDB: {kb_stats['session_docs']} sessions, "
        f"LLM: {'yes (' + enrichment_model + ')' if ollama_available else 'no (heuristics only)'}"
    )

    # Run pipeline
    report = await indexer.run(force=force)

    logger.info("Session indexing complete")
    return report


def _write_report(result: dict):
    """Write indexing report to disk."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = report_dir / "session_index_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Latest summary (text)
    summary_path = LOG_DIR / "latest.txt"
    timestamp = result.get("timestamp", date)
    errors = result.get("errors", [])
    status = "PASS" if not errors else "FAIL"
    latency = result.get("total_latency_ms", 0)

    lines = [
        f"Session Index ({timestamp})",
        "=" * 50,
        f"Status: {status}",
        f"Total time: {latency:.0f}ms",
        "",
        f"Files scanned: {result.get('files_scanned', 0)}",
        f"Files indexed: {result.get('files_indexed', 0)}",
        f"Total sessions: {result.get('total_sessions', 0)}",
        "",
        f"Decisions extracted: {result.get('decisions_added', 0)}",
        f"Lessons extracted: {result.get('lessons_added', 0)}",
        f"Patterns extracted: {result.get('patterns_added', 0)}",
    ]

    if errors:
        lines.append(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            lines.append(f"  - {e}")

    lines.extend(["", f"Reports: {report_dir}"])

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    # Symlink latest report dir
    latest_link = LOG_DIR / "latest-report"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(report_dir)

    logger.info(f"Report written: {report_path}")
    logger.info(f"Summary: {summary_path}")

    # Print summary to stdout
    print("\n".join(lines))

    # Cleanup old reports (keep 30 days)
    import shutil

    for d in LOG_DIR.iterdir():
        if d.is_dir() and d.name.startswith("20") and len(d.name) == 10:
            try:
                age = (datetime.utcnow() - datetime.strptime(d.name, "%Y-%m-%d")).days
                if age > 30:
                    shutil.rmtree(d)
                    logger.info(f"Cleaned old report: {d.name}")
            except ValueError:
                pass


if __name__ == "__main__":
    sys.exit(main())
