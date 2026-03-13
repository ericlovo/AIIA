"""
Backfill Runner — Bulk push local AIIA memories to Supermemory cloud.

One-time catch-up script that syncs all local memories to cloud backup.
No quality gate — everything in memory is already curated.
SHA256 custom_id ensures safe re-runs (Supermemory upserts on custom_id).

Usage:
    python -m local_brain.scripts.backfill_runner [--dry-run] [--categories decisions,patterns]
    # Or from brain CLI: brain backfill [--dry-run] [--categories decisions,patterns]

Reports to ~/.aiia/logs/backfill/
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging before imports that use it
LOG_DIR = Path.home() / ".aiia" / "logs" / "backfill"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "backfill.log"),
    ],
)
logger = logging.getLogger("aiia.backfill_runner")

ALL_CATEGORIES = [
    "decisions",
    "patterns",
    "lessons",
    "sessions",
    "team",
    "agents",
    "meta",
    "project",
    "wip",
]

RATE_LIMIT_MS = 50  # 50ms between API calls


def main():
    # Parse args
    dry_run = "--dry-run" in sys.argv
    categories = None
    for arg in sys.argv[1:]:
        if arg.startswith("--categories"):
            # Handle --categories=x,y or --categories x,y
            if "=" in arg:
                categories = arg.split("=", 1)[1].split(",")
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    categories = sys.argv[idx + 1].split(",")

    if categories:
        # Validate
        invalid = [c for c in categories if c not in ALL_CATEGORIES]
        if invalid:
            print(f"Invalid categories: {invalid}")
            print(f"Valid: {ALL_CATEGORIES}")
            return 1

    logger.info(
        f"Backfill starting — dry_run={dry_run}, categories={categories or 'all'}"
    )

    # Load .env if running standalone
    env_path = Path.home() / ".aiia" / ".env"
    if env_path.exists():
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
        result = asyncio.run(_run_backfill(dry_run, categories))
        _write_report(result)
        return 0 if not result.get("errors") else 1
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        return 1


async def _run_backfill(dry_run: bool, categories: list = None) -> dict:
    """Execute the backfill pipeline."""
    from local_brain.config import get_config
    from local_brain.eq_brain.memory import Memory
    from local_brain.eq_brain.supermemory_bridge import (
        SupermemoryBridge,
    )
    from local_brain.eq_brain.memory_sync import (
        TokenLedger,
        DEFAULT_MONTHLY_BUDGET,
        TOKENS_PER_MEMORY,
    )

    config = get_config()
    memory = Memory(data_dir=config.eq_brain_data_dir)
    bridge = SupermemoryBridge(timeout=config.supermemory_timeout)

    # Token ledger for budget enforcement
    ledger_path = os.path.join(config.eq_brain_data_dir, "sync", "token_ledger.json")
    ledger = TokenLedger(
        ledger_path=ledger_path,
        daily_budget=config.sync_daily_budget,
        monthly_budget=DEFAULT_MONTHLY_BUDGET,
    )

    start = time.monotonic()
    target_categories = categories or ALL_CATEGORIES

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "categories": target_categories,
        "by_category": {},
        "total_synced": 0,
        "total_skipped": 0,
        "total_already_synced": 0,
        "total_errors": 0,
        "tokens_used": 0,
        "errors": [],
    }

    if not dry_run and not bridge.available:
        report["errors"].append("Supermemory bridge unavailable")
        logger.error("Backfill aborted: bridge unavailable")
        return report

    # Load sync state for dedup
    sync_state_path = os.path.join(config.eq_brain_data_dir, "sync", "sync_state.json")
    sync_state = {}
    if os.path.exists(sync_state_path):
        try:
            with open(sync_state_path) as f:
                sync_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    synced_hashes = sync_state.get("synced_hashes", {})

    import hashlib

    for category in target_categories:
        items = memory._read_category(category)
        cat_stats = {
            "total": len(items),
            "synced": 0,
            "skipped": 0,
            "already_synced": 0,
            "errors": 0,
        }

        for item in items:
            fact = item.get("fact", "")
            if not fact:
                cat_stats["skipped"] += 1
                continue

            # Dedup check via content hash
            content_hash = hashlib.sha256(fact.encode()).hexdigest()[:16]
            if content_hash in synced_hashes:
                cat_stats["already_synced"] += 1
                continue

            # Budget check
            if not dry_run and not ledger.can_spend(TOKENS_PER_MEMORY):
                msg = f"Budget exhausted at category [{category}]"
                report["errors"].append(msg)
                logger.warning(msg)
                break

            if dry_run:
                cat_stats["synced"] += 1
                continue

            # Sync to cloud
            result = await bridge.sync_memory(
                fact=fact,
                category=category,
                source=item.get("source", "backfill"),
                metadata={
                    **item.get("metadata", {}),
                    "sync_mode": "backfill",
                },
            )

            if result.get("synced"):
                cat_stats["synced"] += 1
                report["tokens_used"] += TOKENS_PER_MEMORY
                ledger.record_spend(TOKENS_PER_MEMORY, synced=1)

                # Track in sync state
                synced_hashes[content_hash] = {
                    "category": category,
                    "synced_at": datetime.utcnow().isoformat(),
                    "preview": fact[:80],
                }
            else:
                cat_stats["errors"] += 1
                reason = result.get("reason", "unknown")
                report["errors"].append(f"[{category}] {reason}: {fact[:60]}")

            # Rate limit
            await asyncio.sleep(RATE_LIMIT_MS / 1000.0)

        report["by_category"][category] = cat_stats
        report["total_synced"] += cat_stats["synced"]
        report["total_skipped"] += cat_stats["skipped"]
        report["total_already_synced"] += cat_stats["already_synced"]
        report["total_errors"] += cat_stats["errors"]

        logger.info(
            f"[{category}] total={cat_stats['total']} synced={cat_stats['synced']} "
            f"already={cat_stats['already_synced']} skipped={cat_stats['skipped']} "
            f"errors={cat_stats['errors']}"
        )

    # Save sync state (unless dry run)
    if not dry_run:
        sync_state["synced_hashes"] = synced_hashes
        sync_state["last_backfill"] = datetime.utcnow().isoformat()
        os.makedirs(os.path.dirname(sync_state_path), exist_ok=True)
        with open(sync_state_path, "w") as f:
            json.dump(sync_state, f, indent=2, default=str)

    elapsed = (time.monotonic() - start) * 1000
    report["total_latency_ms"] = round(elapsed, 1)
    report["budget_remaining_daily"] = ledger.daily_remaining()
    report["budget_remaining_monthly"] = ledger.monthly_remaining()

    logger.info(
        f"Backfill {'(DRY RUN) ' if dry_run else ''}complete: "
        f"{report['total_synced']} synced, {report['total_already_synced']} already synced, "
        f"{report['total_errors']} errors, {report['tokens_used']} tokens in {elapsed:.0f}ms"
    )

    return report


def _write_report(result: dict):
    """Write backfill report to disk."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = report_dir / "backfill_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Latest summary (text)
    summary_path = LOG_DIR / "latest.txt"
    dry_run = result.get("dry_run", False)
    errors = result.get("errors", [])
    status = "DRY RUN" if dry_run else ("PASS" if not errors else "FAIL")
    latency = result.get("total_latency_ms", 0)

    lines = [
        f"Memory Backfill ({result.get('timestamp', date)})",
        "=" * 50,
        f"Status: {status}",
        f"Total time: {latency:.0f}ms",
        "",
    ]

    for cat, stats in result.get("by_category", {}).items():
        lines.append(
            f"  [{cat}] total={stats['total']} synced={stats['synced']} "
            f"already={stats['already_synced']} skipped={stats['skipped']} "
            f"errors={stats['errors']}"
        )

    lines.extend(
        [
            "",
            f"Total synced: {result.get('total_synced', 0)}",
            f"Already synced: {result.get('total_already_synced', 0)}",
            f"Tokens used: {result.get('tokens_used', 0):,}",
            f"Budget remaining (daily): {result.get('budget_remaining_daily', 0):,}",
            f"Budget remaining (monthly): {result.get('budget_remaining_monthly', 0):,}",
        ]
    )

    if errors:
        lines.append(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
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

    # Print summary to stdout
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
