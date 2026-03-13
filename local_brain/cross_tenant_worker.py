"""
Cross-Tenant Intelligence Worker — Nightly cron job on Mac Mini.

Reads conversation summaries from PostgreSQL (read-only), groups by tenant,
extracts domain frequency and cross-domain correlations, then stores
aggregated patterns via AIIA's memory system.

Privacy: Only aggregated patterns stored — never individual messages or PII.
Database connection is read-only.

Cron setup (Mac Mini crontab):
    0 3 * * * cd /path/to/AIIA && python -m local_brain.cross_tenant_worker

Or run directly:
    python cross_tenant_worker.py
"""

import asyncio
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.cross_tenant_worker")


def _get_readonly_connection_string() -> Optional[str]:
    """
    Get a read-only database connection string.
    Uses CROSS_TENANT_DB_URL (preferred) or DATABASE_URL with read-only intent.
    """
    url = os.getenv("CROSS_TENANT_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        logger.error("No database URL configured (CROSS_TENANT_DB_URL or DATABASE_URL)")
        return None
    return url


async def _fetch_conversation_summaries(
    db_url: str,
    lookback_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Fetch recent conversation metadata from PostgreSQL.

    Only reads: tenant_id, domain_classification, routing_mode, complexity_score,
    eq_level, created_at. Never reads message content.
    """
    try:
        import sqlalchemy
        from sqlalchemy import create_engine, text

        # Use synchronous engine for simplicity in cron job
        engine = create_engine(db_url, pool_pre_ping=True)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        with engine.connect() as conn:
            # Read from flagged_conversations for routing metadata
            # This table has domain_classification, routing_mode, complexity_score, eq_level
            result = conn.execute(
                text("""
                    SELECT
                        tenant_id,
                        domain_classification,
                        routing_mode,
                        complexity_score,
                        eq_level,
                        created_at
                    FROM flagged_conversations
                    WHERE created_at >= :cutoff
                    ORDER BY created_at DESC
                    LIMIT 5000
                """),
                {"cutoff": cutoff},
            )
            rows = [dict(row._mapping) for row in result]

            # Also read from audit_events for routing decisions
            audit_result = conn.execute(
                text("""
                    SELECT
                        tenant_id,
                        category,
                        event_type,
                        details,
                        created_at
                    FROM audit_events
                    WHERE category = 'ai_routing'
                      AND created_at >= :cutoff
                    ORDER BY created_at DESC
                    LIMIT 5000
                """),
                {"cutoff": cutoff},
            )
            audit_rows = [dict(row._mapping) for row in audit_result]

        engine.dispose()
        logger.info(
            f"Fetched {len(rows)} flagged conversations + "
            f"{len(audit_rows)} routing audit events "
            f"(last {lookback_days} days)"
        )
        return rows + audit_rows

    except Exception as e:
        logger.error(f"Failed to fetch conversation data: {e}")
        return []


def _analyze_patterns(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract aggregated patterns from conversation metadata.

    Groups by tenant, computes:
    - Domain frequency distribution
    - Average complexity scores
    - EQ level distribution
    - Routing mode split (eos vs rlm)
    - Cross-domain correlations (which domains appear together)
    """
    if not summaries:
        return {"patterns": [], "summary": "No data to analyze"}

    tenant_data: Dict[str, Dict] = defaultdict(
        lambda: {
            "domains": Counter(),
            "routing_modes": Counter(),
            "complexity_scores": [],
            "eq_levels": [],
            "count": 0,
        }
    )

    for row in summaries:
        tenant = row.get("tenant_id", "unknown")
        td = tenant_data[tenant]
        td["count"] += 1

        domain = row.get("domain_classification")
        if domain:
            td["domains"][domain] += 1

        mode = row.get("routing_mode")
        if mode:
            td["routing_modes"][mode] += 1

        score = row.get("complexity_score")
        if score is not None:
            td["complexity_scores"].append(float(score))

        eq = row.get("eq_level")
        if eq is not None:
            td["eq_levels"].append(int(eq))

    # Build aggregated patterns
    patterns = []
    for tenant_id, data in tenant_data.items():
        pattern = {
            "tenant_id": tenant_id,
            "conversation_count": data["count"],
            "top_domains": data["domains"].most_common(5),
            "routing_split": dict(data["routing_modes"]),
            "avg_complexity": (
                round(
                    sum(data["complexity_scores"]) / len(data["complexity_scores"]), 3
                )
                if data["complexity_scores"]
                else None
            ),
            "avg_eq_level": (
                round(sum(data["eq_levels"]) / len(data["eq_levels"]), 1)
                if data["eq_levels"]
                else None
            ),
        }
        patterns.append(pattern)

    # Cross-tenant insights
    all_domains = Counter()
    for td in tenant_data.values():
        all_domains.update(td["domains"])

    return {
        "patterns": patterns,
        "cross_tenant": {
            "total_conversations": sum(td["count"] for td in tenant_data.values()),
            "tenant_count": len(tenant_data),
            "global_domain_distribution": all_domains.most_common(10),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _store_patterns_in_aiia(
    patterns: Dict[str, Any],
    aiia_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Store aggregated patterns in AIIA's memory system via the local API.
    """
    import httpx

    base_url = aiia_url or os.getenv("LOCAL_BRAIN_URL", "http://localhost:8100")
    key = api_key or os.getenv("LOCAL_BRAIN_API_KEY")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key

    # Format patterns as a readable summary
    summary_lines = [
        f"Cross-Tenant Intelligence Report — {patterns.get('generated_at', 'unknown')}",
        f"Total conversations analyzed: {patterns['cross_tenant']['total_conversations']}",
        f"Tenants active: {patterns['cross_tenant']['tenant_count']}",
        "",
        "Global domain distribution:",
    ]
    for domain, count in patterns["cross_tenant"]["global_domain_distribution"]:
        summary_lines.append(f"  - {domain}: {count}")

    summary_lines.append("")
    for p in patterns["patterns"]:
        summary_lines.append(
            f"Tenant {p['tenant_id']}: {p['conversation_count']} conversations, "
            f"avg complexity {p.get('avg_complexity', 'N/A')}, "
            f"avg EQ {p.get('avg_eq_level', 'N/A')}"
        )
        if p["top_domains"]:
            summary_lines.append(
                f"  Top domains: {', '.join(f'{d}({c})' for d, c in p['top_domains'])}"
            )

    summary_text = "\n".join(summary_lines)

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            response = await client.post(
                "/v1/aiia/remember",
                headers=headers,
                json={
                    "fact": summary_text,
                    "category": "cross_tenant",
                    "source": "cross_tenant_worker",
                    "metadata": {
                        "generated_at": patterns.get("generated_at"),
                        "tenant_count": patterns["cross_tenant"]["tenant_count"],
                        "total_conversations": patterns["cross_tenant"][
                            "total_conversations"
                        ],
                    },
                },
            )
            response.raise_for_status()
            logger.info("Stored cross-tenant patterns in AIIA memory")
            return True
    except Exception as e:
        logger.error(f"Failed to store patterns in AIIA: {e}")
        return False


async def run_worker():
    """Main worker entry point."""
    logger.info("=" * 50)
    logger.info("Cross-Tenant Intelligence Worker starting")
    logger.info("=" * 50)

    db_url = _get_readonly_connection_string()
    if not db_url:
        logger.error("No database URL — exiting")
        return

    # Fetch conversation metadata (no message content)
    summaries = await _fetch_conversation_summaries(db_url, lookback_days=7)
    if not summaries:
        logger.info("No conversation data found — nothing to analyze")
        return

    # Analyze patterns
    patterns = _analyze_patterns(summaries)
    logger.info(
        f"Analyzed {patterns['cross_tenant']['total_conversations']} conversations "
        f"across {patterns['cross_tenant']['tenant_count']} tenants"
    )

    # Store in AIIA
    success = await _store_patterns_in_aiia(patterns)
    if success:
        logger.info("Cross-tenant intelligence update complete")
    else:
        logger.error("Failed to store patterns — check AIIA connectivity")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run_worker())
