"""
Research Runner — CLI entry point for the autonomous research loop.

Ensures the seed topics exist (see research/seeds.py), then runs one
research cycle via ResearchScheduler: N sessions across the most-stale
active topics. Designed to be invoked nightly by a launchd/cron agent, the
same way consolidation_runner.py is.

Usage:
    python -m local_brain.scripts.research_runner [--force] [--list]
                                                  [--max-topics N] [--sessions N]

    --force        run even if AIIA_RESEARCH_ENABLED / autonomy level gate is off
    --list         list active topics and exit (no sessions run)
    --max-topics   override AIIA_RESEARCH_MAX_TOPICS for this run
    --sessions     override AIIA_RESEARCH_SESSIONS_PER_TOPIC for this run

Schedule:
    Nightly via a com.aiia.research launchd agent (operator-installed).

Gating:
    Honors AutonomyConfig (level=phase2 + research_enabled). --force bypasses
    the gate for manual/operator runs.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path.home() / ".aiia" / "logs" / "research"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "research.log"),
    ],
)
logger = logging.getLogger("aiia.research_runner")


def _load_env() -> None:
    env_path = Path.home() / ".aiia" / ".env"
    if not env_path.exists():
        return
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


def _parse_int_flag(name: str) -> int | None:
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv):
            try:
                return int(sys.argv[idx + 1])
            except ValueError:
                logger.warning("Ignoring non-integer value for %s", name)
    return None


def main() -> int:
    force = "--force" in sys.argv
    list_only = "--list" in sys.argv
    max_topics = _parse_int_flag("--max-topics")
    sessions = _parse_int_flag("--sessions")

    _load_env()
    repo_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_dir))

    try:
        return asyncio.run(_run(force, list_only, max_topics, sessions))
    except Exception as e:
        logger.error("Research run failed: %s", e, exc_info=True)
        return 1


async def _run(force: bool, list_only: bool, max_topics: int | None, sessions: int | None) -> int:
    from local_brain.autonomy.research_loop import ResearchScheduler
    from local_brain.config import AutonomyConfig, get_config
    from local_brain.eq_brain.knowledge_store import KnowledgeStore
    from local_brain.ollama_client import OllamaClient
    from local_brain.research.engine import ResearchEngine
    from local_brain.research.seeds import ensure_seed_topics
    from local_brain.research.topic import TopicStore

    config = get_config()
    store = TopicStore(data_dir=config.eq_brain_data_dir)

    created = ensure_seed_topics(store)
    if created:
        logger.info("Seeded %d new topic(s): %s", len(created), ", ".join(t.title for t in created))

    active = [t for t in store.list_all() if t.status == "active"]
    logger.info("Active topics: %d", len(active))
    if list_only:
        for t in active:
            logger.info("  [%s] %s (runs=%d, last=%s)", t.id, t.title, t.run_count, t.last_run)
        return 0

    autonomy = AutonomyConfig.from_env()
    if force:
        autonomy.level = "phase2"
        autonomy.research_enabled = True
    if max_topics is not None:
        autonomy.research_max_topics_per_cycle = max_topics
    if sessions is not None:
        autonomy.research_sessions_per_topic = sessions

    if not (autonomy.level == "phase2" and autonomy.research_enabled):
        logger.info(
            "Research loop disabled (level=%s, research_enabled=%s). "
            "Set AIIA_AUTONOMY_LEVEL=phase2 + AIIA_RESEARCH_ENABLED=true, or pass --force.",
            autonomy.level,
            autonomy.research_enabled,
        )
        return 0

    ollama = OllamaClient(config=config)
    health = await ollama.health()
    if health.get("status") != "online":
        logger.error("Ollama offline: %s", health.get("error", "unknown"))
        return 1

    knowledge = KnowledgeStore(
        data_dir=config.eq_brain_data_dir,
        collection_name=config.eq_brain_collection,
    )
    await knowledge.initialize()

    task_model = config.models.get("task")
    model = task_model.model_name if task_model else "llama3.1:8b-instruct-q8_0"

    engine = ResearchEngine(
        ollama=ollama,
        knowledge=knowledge,
        topic_store=store,
        model=model,
    )

    scheduler = ResearchScheduler(config=autonomy, engine=engine, topic_store=store)
    result = await scheduler.run_cycle()

    logger.info(
        "Research cycle done: worked=%d sessions=%d errors=%d",
        result.get("topics_worked", 0),
        result.get("sessions_run", 0),
        result.get("errors", 0),
    )
    for t in result.get("topics", []):
        logger.info("  [%s] %s — %d session(s)", t["id"], t["title"], t["sessions"])

    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
