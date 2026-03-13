"""
Morning Briefing Runner — CLI entry point for the smart alerting pipeline.

Reads overnight reports (security, sync, code), feeds to DeepSeek R1
for severity-ranked synthesis, creates ActionQueue items for critical findings.

Usage:
    python -m local_brain.scripts.morning_briefing_runner
    # Or from brain CLI: brain briefing

Schedule:
    Daily at 4:30am via com.aiia.briefing launchd agent
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging before imports that use it
LOG_DIR = Path.home() / ".aiia" / "logs" / "briefings"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "briefing.log"),
    ],
)
logger = logging.getLogger("aiia.briefing_runner")


def main():
    logger.info("Morning briefing starting")

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
        result = asyncio.run(_run_briefing())
        _write_report(result)
        return 0 if not result.get("errors") else 1
    except Exception as e:
        logger.error(f"Morning briefing failed: {e}", exc_info=True)
        return 1


async def _run_briefing() -> dict:
    """Run the morning briefing pipeline."""
    from local_brain.command_center.action_queue import ActionQueue
    from local_brain.config import get_config
    from local_brain.eq_brain.brain import AIIA
    from local_brain.eq_brain.knowledge_store import KnowledgeStore
    from local_brain.eq_brain.memory import Memory
    from local_brain.eq_brain.morning_briefing import MorningBriefing
    from local_brain.ollama_client import OllamaClient

    config = get_config()

    # Check Ollama health
    ollama = OllamaClient(config=config)
    health = await ollama.health()
    if health["status"] != "online":
        logger.error(f"Ollama offline: {health.get('error', 'unknown')}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "errors": ["Ollama offline — cannot run briefing"],
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
        logger.warning(
            "No deep model configured — briefing will use heuristic fallback"
        )

    aiia = AIIA(
        knowledge_store=knowledge,
        memory=memory,
        ollama_client=ollama,
        model=model,
        deep_model=deep_model,
    )

    # Initialize ActionQueue
    action_queue = ActionQueue()

    # Expire old pending actions (72h)
    expired = action_queue.expire_old()
    if expired:
        logger.info(f"Expired {expired} stale actions")

    # Log state
    mem_stats = memory.stats()
    action_summary = action_queue.summary()
    logger.info(
        f"Memory: {mem_stats['total_memories']} total, "
        f"Actions: {action_summary['total']} total ({action_summary.get('by_status', {}).get('pending', 0)} pending)"
    )

    # Generate briefing
    briefing = MorningBriefing(
        aiia=aiia,
        memory=memory,
        action_queue=action_queue,
    )

    report = await briefing.generate()
    return report


def _write_report(result: dict):
    """Write briefing report to disk."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = report_dir / "briefing_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Latest summary (text)
    summary_path = LOG_DIR / "latest.txt"
    timestamp = result.get("timestamp", date)
    grade = result.get("health_grade", "?")
    alerts = result.get("alerts", [])
    summary = result.get("summary", "No summary")
    latency = result.get("latency_ms", 0)
    actions = result.get("actions_created", 0)
    fallback = result.get("fallback", False)

    lines = [
        f"Morning Briefing ({timestamp})",
        "=" * 50,
        f"Health Grade: {grade}",
        f"Model: {'heuristic fallback' if fallback else 'DeepSeek R1'}",
        f"Time: {latency:.0f}ms",
        "",
        f"Summary: {summary}",
        "",
    ]

    if alerts:
        lines.append(f"Alerts ({len(alerts)}):")
        for alert in alerts:
            sev = alert.get("severity", "?").upper()
            title = alert.get("title", "?")
            detail = alert.get("detail", "")
            lines.append(f"  [{sev}] {title}")
            if detail:
                lines.append(f"         {detail}")
    else:
        lines.append("Alerts: None")

    code_notes = result.get("code_review_notes", [])
    if code_notes:
        lines.append(f"\nCode Review Notes ({len(code_notes)}):")
        for note in code_notes:
            sev = note.get("severity", "info").upper()
            text = note.get("note", "?")
            lines.append(f"  [{sev}] {text}")

    if actions:
        lines.append(f"\nActionQueue: {actions} new items created")

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

    # Print human-readable briefing to stdout
    print()
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
