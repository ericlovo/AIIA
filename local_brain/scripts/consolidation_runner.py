"""
Memory Consolidation Runner — CLI entry point for nightly consolidation.

Uses DeepSeek R1 (14b) to analyze accumulated memories, find themes,
detect contradictions, and flag outdated entries.

Usage:
    python -m local_brain.scripts.consolidation_runner [--force] [--category X]
    # Or from brain CLI: brain consolidate [--force] [--category X]

Schedule:
    Daily at 3:00am via com.aiia.consolidate launchd agent
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging before imports that use it
LOG_DIR = Path.home() / ".aiia" / "logs" / "consolidation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "consolidation.log"),
    ],
)
logger = logging.getLogger("aiia.consolidation_runner")


def main():
    # Parse args
    force = "--force" in sys.argv
    category = None
    if "--category" in sys.argv:
        idx = sys.argv.index("--category")
        if idx + 1 < len(sys.argv):
            category = sys.argv[idx + 1]

    logger.info(
        f"Consolidation starting — force={force}"
        + (f", category={category}" if category else "")
    )

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
        result = asyncio.run(_run_consolidation(force, category))
        _write_report(result)
        return 0 if not result.get("errors") else 1
    except Exception as e:
        logger.error(f"Consolidation failed: {e}", exc_info=True)
        return 1


async def _run_consolidation(force: bool, category: str = None) -> dict:
    """Run the consolidation pipeline."""
    from local_brain.config import get_config
    from local_brain.eq_brain.brain import AIIA
    from local_brain.eq_brain.knowledge_store import KnowledgeStore
    from local_brain.eq_brain.memory import Memory
    from local_brain.eq_brain.memory_consolidator import (
        MemoryConsolidator,
    )
    from local_brain.ollama_client import OllamaClient

    config = get_config()

    # Check Ollama health first
    ollama = OllamaClient(config=config)
    health = await ollama.health()
    if health["status"] != "online":
        logger.error(f"Ollama offline: {health.get('error', 'unknown')}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "errors": ["Ollama offline — cannot run consolidation"],
        }

    # Initialize AIIA with deep model
    memory = Memory(data_dir=config.eq_brain_data_dir)
    knowledge = KnowledgeStore(
        data_dir=config.eq_brain_data_dir,
        collection_name=config.eq_brain_collection,
    )
    await knowledge.initialize()

    task_model = config.models.get("task")
    model = task_model.model_name if task_model else "llama3.1:8b-instruct-q8_0"
    deep_cfg = config.models.get("deep")
    deep_model = deep_cfg.model_name if deep_cfg else None

    if not deep_model:
        logger.error("No deep model configured")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "errors": ["No deep model configured — set LOCAL_DEEP_MODEL"],
        }

    aiia = AIIA(
        knowledge_store=knowledge,
        memory=memory,
        ollama_client=ollama,
        model=model,
        deep_model=deep_model,
    )

    # Log pre-run state
    mem_stats = memory.stats()
    logger.info(
        f"Memory: {mem_stats['total_memories']} total, deep model: {deep_model}"
    )

    # Run consolidation
    consolidator = MemoryConsolidator(
        aiia=aiia,
        memory=memory,
        data_dir=config.eq_brain_data_dir,
    )

    categories = [category] if category else None
    report = await consolidator.run_all(categories=categories, force=force)

    logger.info("Consolidation complete")
    return report


def _write_report(result: dict):
    """Write consolidation report to disk."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = report_dir / "consolidation_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Latest summary (text)
    summary_path = LOG_DIR / "latest.txt"
    timestamp = result.get("timestamp", date)
    errors = result.get("errors", [])
    status = "PASS" if not errors else "FAIL"
    latency = result.get("total_latency_ms", 0)

    lines = [
        f"Memory Consolidation ({timestamp})",
        "=" * 50,
        f"Status: {status}",
        f"Total time: {latency:.0f}ms",
        "",
    ]

    categories = result.get("categories", {})
    for cat_name, cat_result in categories.items():
        if cat_result.get("skipped"):
            lines.append(f"  {cat_name}: skipped ({cat_result.get('reason', '?')})")
        elif cat_result.get("error"):
            lines.append(f"  {cat_name}: ERROR ({cat_result['error']})")
        else:
            stats = cat_result.get("stats", {})
            lines.append(
                f"  {cat_name}: {stats.get('themes_found', 0)} themes, "
                f"{stats.get('contradictions_found', 0)} contradictions, "
                f"{stats.get('outdated_found', 0)} outdated "
                f"({cat_result.get('latency_ms', 0):.0f}ms)"
            )

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
