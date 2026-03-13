"""
Metered Memory Sync Runner — CLI entry point for the sync pipeline.

Instantiates Memory, SupermemoryBridge, OllamaClient, TokenLedger, and MeteredSync,
runs the sync, and outputs the report.

Usage:
    python -m local_brain.scripts.memory_sync_runner [--weekly]
    # Or from brain CLI: brain sync [--weekly]

Schedule:
    Daily at 1am via com.aiia.memorysync launchd agent
    Weekly mode runs on Sundays (includes Tier 2 categories)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging before imports that use it
LOG_DIR = Path.home() / ".aiia" / "logs" / "sync"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "sync.log"),
    ],
)
logger = logging.getLogger("aiia.memory_sync_runner")


def main():
    # Parse args
    weekly = "--weekly" in sys.argv or "-w" in sys.argv

    # Auto-detect weekly: if today is Sunday, run weekly mode
    if not weekly and datetime.utcnow().weekday() == 6:
        weekly = True
        logger.info("Sunday detected — running weekly mode (includes Tier 2)")

    logger.info(f"Memory sync starting — mode={'weekly' if weekly else 'daily'}")

    # Load .env if running standalone (not via start_brain.sh)
    env_path = Path.home() / ".aiia" / ".env"
    if env_path.exists() and not os.getenv("SUPERMEMORY_API_KEY"):
        logger.info("Loading .env for standalone execution")
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
        result = asyncio.run(_run_sync(weekly))
        _write_report(result)
        return 0 if not result.get("errors") else 1
    except Exception as e:
        logger.error(f"Memory sync failed: {e}", exc_info=True)
        return 1


async def _run_sync(weekly: bool) -> dict:
    """Run the metered sync pipeline."""
    from local_brain.config import get_config
    from local_brain.eq_brain.memory import Memory
    from local_brain.eq_brain.memory_sync import (
        MeteredSync,
        TokenLedger,
    )
    from local_brain.eq_brain.supermemory_bridge import (
        SupermemoryBridge,
    )
    from local_brain.ollama_client import OllamaClient

    config = get_config()

    # Initialize components
    memory = Memory(data_dir=config.eq_brain_data_dir)
    bridge = SupermemoryBridge(timeout=config.supermemory_timeout)
    ollama = OllamaClient(config=config)

    # Token ledger persists in eq_data/sync/
    sync_dir = os.path.join(config.eq_brain_data_dir, "sync")
    os.makedirs(sync_dir, exist_ok=True)
    ledger = TokenLedger(
        ledger_path=os.path.join(sync_dir, "token_ledger.json"),
        daily_budget=config.sync_daily_budget,
    )

    # Log pre-sync state
    mem_stats = memory.stats()
    bridge_status = bridge.status()
    ledger_status = ledger.status()
    logger.info(f"Memory stats: {mem_stats['total_memories']} total memories")
    logger.info(
        f"Bridge: available={bridge_status['available']}, api_key_set={bridge_status['api_key_set']}"
    )
    logger.info(
        f"Budget: {ledger_status['daily_remaining']:,} daily / {ledger_status['monthly_remaining']:,} monthly tokens remaining"
    )

    # Check Ollama health
    ollama_health = await ollama.health()
    if ollama_health["status"] != "online":
        logger.error(f"Ollama offline: {ollama_health.get('error', 'unknown')}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "errors": ["Ollama offline — cannot score memories"],
            "mode": "weekly" if weekly else "daily",
        }

    # Get scorer model name from config
    scorer_model = config.models.get("task", config.models.get("routing"))
    model_name = (
        scorer_model.model_name if scorer_model else "llama3.1:8b-instruct-q8_0"
    )

    # Parse project excluded sources from config
    excluded = set(
        s.strip() for s in config.sync_project_excluded_sources.split(",") if s.strip()
    )

    # Run metered sync
    sync = MeteredSync(
        memory=memory,
        supermemory_bridge=bridge,
        ollama_client=ollama,
        ledger=ledger,
        scorer_model=model_name,
        quality_gate=config.sync_quality_gate,
        project_excluded_sources=excluded,
    )

    report = await sync.run(include_tier2=weekly)

    # Log summary
    logger.info(report.summary_text())

    return report.to_dict()


def _write_report(result: dict):
    """Write sync report to disk."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = report_dir / "sync_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Latest summary (text)
    summary_path = LOG_DIR / "latest.txt"
    mode = result.get("mode", "daily")
    timestamp = result.get("timestamp", date)
    errors = result.get("errors", [])
    status = "PASS" if not errors else "FAIL"

    t1 = result.get("tier1", {})
    t2 = result.get("tier2", {})
    tokens = result.get("tokens", {})
    scoring = result.get("scoring", {})

    lines = [
        f"Memory Sync — {mode.title()} ({timestamp})",
        "=" * 50,
        f"Status: {status}",
        "",
        "Tier 1 (decisions, patterns, lessons, sessions):",
        f"  Synced: {t1.get('synced', 0)}  Skipped: {t1.get('skipped', 0)}  Already synced: {t1.get('already_synced', 0)}",
    ]
    if mode == "weekly":
        lines.extend(
            [
                "Tier 2 (team, agents, meta):",
                f"  Synced: {t2.get('synced', 0)}  Skipped: {t2.get('skipped', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            f"Quality Scoring: {scoring.get('memories_scored', 0)} evaluated, avg {scoring.get('avg_score', 0):.1f}/5",
            f"Tokens Used: {tokens.get('used', 0):,}",
            f"Daily Remaining: {tokens.get('remaining_daily', 0):,}",
            f"Monthly Remaining: {tokens.get('remaining_monthly', 0):,}",
        ]
    )
    if result.get("decayed"):
        lines.append(f"Decayed: {result['decayed']}")
    if errors:
        lines.append(f"Errors ({len(errors)}):")
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

    cutoff = datetime.utcnow().strftime("%Y-%m-%d")
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
