"""
Metered Memory Sync — Tiered, quality-scored, budget-aware cloud synchronization.

The bridge between AIIA's local memory and Supermemory cloud, governed by
token economics and memory quality. Not everything deserves to be remembered
forever — this module decides what does.

Architecture:
    TokenLedger   — Daily/monthly token budget tracking with hard caps
    MemoryScorer  — Local LLM ($0) evaluates memory quality (1-5 scale)
    MeteredSync   — Orchestrator: tier classification, quality filter,
                    metered push, decay, consolidation

Tier System:
    Tier 1 (daily):  decisions, patterns, lessons, sessions — high signal
    Tier 2 (weekly): team, agents, meta, project — moderate signal, cloud-backed
    Tier 3 (never):  wip (ephemeral only)

Token Budget:
    3M tokens/month from Supermemory
    ~200K tokens/day allocation (burst capacity for catch-up)
    Each memory sync ≈ 150-200 tokens (content + embedding overhead)
    Quality gate: memories scoring 3+ get synced (env: SYNC_QUALITY_GATE)

Schedule:
    Daily at 1am via com.aiia.memorysync launchd agent
    After security scan (midnight) and before morning work
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.eq_brain.memory_sync")

# ═══════════════════════════════════════════════════════════════════
# Tier Classification
# ═══════════════════════════════════════════════════════════════════

TIER_1_CATEGORIES = ["decisions", "patterns", "lessons", "sessions"]
TIER_2_CATEGORIES = ["team", "agents", "meta", "project"]
TIER_3_CATEGORIES = ["wip"]  # Never auto-synced (ephemeral only)

# Default budget: 3M tokens/month, ~200K/day (burst capacity for catch-up)
DEFAULT_MONTHLY_BUDGET = 3_000_000
DEFAULT_DAILY_BUDGET = 200_000
TOKENS_PER_MEMORY = 175  # Estimated: content + embedding overhead

# Default sources excluded from project sync — high-volume noise, not worth cloud tokens
# Override via SYNC_PROJECT_EXCLUDED env var (comma-separated)
DEFAULT_PROJECT_EXCLUDED_SOURCES = {
    "health_journal",
    "code_health",
    "test_run",
    "security_scan",
}


# ═══════════════════════════════════════════════════════════════════
# Token Ledger
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DayEntry:
    date: str
    tokens_used: int = 0
    memories_synced: int = 0
    memories_skipped: int = 0
    memories_decayed: int = 0


class TokenLedger:
    """
    Tracks Supermemory token consumption with daily/monthly caps.

    Persists to JSON so budget survives restarts. Designed to prevent
    the overuse that happened before — hard caps, no exceptions.
    """

    def __init__(
        self,
        ledger_path: str,
        daily_budget: int = DEFAULT_DAILY_BUDGET,
        monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
    ):
        self._path = ledger_path
        self._daily_budget = daily_budget
        self._monthly_budget = monthly_budget
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"days": {}, "created": datetime.utcnow().isoformat()}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _today_key(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _month_key(self) -> str:
        return datetime.utcnow().strftime("%Y-%m")

    def _get_day(self, date_key: str) -> Dict[str, int]:
        return self._data["days"].get(
            date_key,
            {
                "tokens_used": 0,
                "memories_synced": 0,
                "memories_skipped": 0,
                "memories_decayed": 0,
            },
        )

    def daily_used(self) -> int:
        return self._get_day(self._today_key()).get("tokens_used", 0)

    def daily_remaining(self) -> int:
        return max(0, self._daily_budget - self.daily_used())

    def monthly_used(self) -> int:
        month = self._month_key()
        total = 0
        for key, day in self._data["days"].items():
            if key.startswith(month):
                total += day.get("tokens_used", 0)
        return total

    def monthly_remaining(self) -> int:
        return max(0, self._monthly_budget - self.monthly_used())

    def can_spend(self, tokens: int) -> bool:
        return self.daily_remaining() >= tokens and self.monthly_remaining() >= tokens

    def record_spend(
        self,
        tokens: int,
        synced: int = 0,
        skipped: int = 0,
        decayed: int = 0,
    ):
        key = self._today_key()
        day = self._get_day(key)
        day["tokens_used"] = day.get("tokens_used", 0) + tokens
        day["memories_synced"] = day.get("memories_synced", 0) + synced
        day["memories_skipped"] = day.get("memories_skipped", 0) + skipped
        day["memories_decayed"] = day.get("memories_decayed", 0) + decayed
        self._data["days"][key] = day
        self._save()

    def status(self) -> Dict[str, Any]:
        return {
            "daily_budget": self._daily_budget,
            "daily_used": self.daily_used(),
            "daily_remaining": self.daily_remaining(),
            "monthly_budget": self._monthly_budget,
            "monthly_used": self.monthly_used(),
            "monthly_remaining": self.monthly_remaining(),
            "today": self._get_day(self._today_key()),
        }

    def cleanup_old_entries(self, keep_days: int = 90):
        cutoff = (datetime.utcnow() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        old_keys = [k for k in self._data["days"] if k < cutoff]
        for k in old_keys:
            del self._data["days"][k]
        if old_keys:
            self._save()
            logger.info(f"Cleaned {len(old_keys)} old ledger entries")


# ═══════════════════════════════════════════════════════════════════
# Memory Quality Scorer
# ═══════════════════════════════════════════════════════════════════

SCORER_PROMPT = """You are a memory quality evaluator for an AI platform.

Score this memory on a scale of 1-5 for long-term persistence value:

1 = Noise. Temporary state, obvious fact, or session-specific chatter.
2 = Low value. Might be useful once but not worth remembering long-term.
3 = Moderate. Useful context but not critical. Could be re-derived if lost.
4 = Valuable. A genuine insight, pattern, or decision that should persist.
5 = Critical. An architecture decision, hard-won lesson, or team agreement
    that would be costly to lose.

Category: {category}
Memory: {fact}

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


class MemoryScorer:
    """
    Uses the local LLM ($0) to evaluate whether a memory is worth
    syncing to Supermemory cloud. This is the quality gate that
    prevents token waste on noise.

    Memories scoring 4+ get synced. Below 4 stays local-only.
    """

    def __init__(
        self,
        ollama_client,
        model: str = "llama3.1:8b-instruct-q8_0",
        min_score: int = 3,
    ):
        self._ollama = ollama_client
        self._model = model
        self._min_score = min_score  # Threshold for cloud sync (env: SYNC_QUALITY_GATE)

    async def score(self, fact: str, category: str) -> Dict[str, Any]:
        """Score a memory's persistence value. Returns {score, reason, worthy}."""
        # Skip scoring for categories that are always high-value
        if category in ("decisions",):
            return {"score": 5, "reason": "decisions always persist", "worthy": True}

        prompt = SCORER_PROMPT.format(category=category, fact=fact[:500])

        try:
            response = await self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Deterministic scoring
                max_tokens=100,
                num_ctx=4096,  # Small context for scoring
            )

            content = response.get("message", {}).get("content", "")

            # Parse JSON from response
            import re

            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                score = int(result.get("score", 3))
                reason = result.get("reason", "no reason given")
            else:
                score = 3
                reason = "could not parse scorer response"

            return {
                "score": min(5, max(1, score)),
                "reason": reason,
                "worthy": score >= self._min_score,
            }

        except Exception as e:
            logger.warning(f"Memory scoring failed, defaulting to worthy: {e}")
            # On scorer failure, default to syncing (safe fallback)
            return {"score": 4, "reason": f"scorer error: {e}", "worthy": True}


# ═══════════════════════════════════════════════════════════════════
# Decay Engine
# ═══════════════════════════════════════════════════════════════════

# How long memories survive in each category before pruning
DECAY_POLICIES = {
    "wip": timedelta(hours=24),
    "sessions": timedelta(days=90),
    "project": timedelta(days=90),  # Extended from 60d — cloud-backed now
    "meta": timedelta(days=180),
    # decisions, patterns, lessons, team, agents = permanent (no decay)
}


def apply_decay(memory, data_dir: str) -> Dict[str, int]:
    """
    Prune expired memories based on decay policies.

    Returns count of memories removed per category.
    """
    from local_brain.eq_brain.memory import Memory

    decayed = {}
    now = datetime.utcnow()

    for category, ttl in DECAY_POLICIES.items():
        items = memory._read_category(category)
        if not items:
            continue

        cutoff = (now - ttl).isoformat()
        original_count = len(items)
        surviving = [item for item in items if item.get("created_at", "9999") > cutoff]

        removed = original_count - len(surviving)
        if removed > 0:
            memory._write_category(category, surviving)
            decayed[category] = removed
            logger.info(f"Decayed {removed} memories from [{category}] (TTL: {ttl})")

    return decayed


# ═══════════════════════════════════════════════════════════════════
# Metered Sync Orchestrator
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SyncReport:
    """Results from a metered sync run."""

    timestamp: str = ""
    mode: str = "daily"  # daily or weekly
    tier1_synced: int = 0
    tier1_skipped: int = 0
    tier1_already_synced: int = 0
    tier2_synced: int = 0
    tier2_skipped: int = 0
    tokens_used: int = 0
    tokens_remaining_daily: int = 0
    tokens_remaining_monthly: int = 0
    memories_scored: int = 0
    avg_score: float = 0.0
    decayed: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    circuit_breaker_tripped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "tier1": {
                "synced": self.tier1_synced,
                "skipped": self.tier1_skipped,
                "already_synced": self.tier1_already_synced,
            },
            "tier2": {
                "synced": self.tier2_synced,
                "skipped": self.tier2_skipped,
            },
            "tokens": {
                "used": self.tokens_used,
                "remaining_daily": self.tokens_remaining_daily,
                "remaining_monthly": self.tokens_remaining_monthly,
            },
            "scoring": {
                "memories_scored": self.memories_scored,
                "avg_score": round(self.avg_score, 2),
            },
            "decayed": self.decayed,
            "errors": self.errors,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
        }

    def summary_text(self) -> str:
        status = (
            "PASS" if not self.errors and not self.circuit_breaker_tripped else "FAIL"
        )
        lines = [
            f"Memory Sync — {self.mode.title()} ({self.timestamp})",
            "=" * 50,
            f"Status: {status}",
            "",
            "Tier 1 (decisions, patterns, lessons, sessions):",
            f"  Synced: {self.tier1_synced}  Skipped: {self.tier1_skipped}  Already synced: {self.tier1_already_synced}",
        ]
        if self.mode == "weekly":
            lines.extend(
                [
                    "Tier 2 (team, agents, meta):",
                    f"  Synced: {self.tier2_synced}  Skipped: {self.tier2_skipped}",
                ]
            )
        lines.extend(
            [
                "",
                f"Quality Scoring: {self.memories_scored} evaluated, avg {self.avg_score:.1f}/5",
                f"Tokens Used: {self.tokens_used:,} / {self.tokens_used + self.tokens_remaining_daily:,} daily",
                f"Monthly Remaining: {self.tokens_remaining_monthly:,}",
            ]
        )
        if self.decayed:
            lines.append(f"Decayed: {self.decayed}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
        return "\n".join(lines)


class MeteredSync:
    """
    Orchestrates tiered, quality-scored, budget-aware memory synchronization
    from AIIA's local memory to Supermemory cloud.

    This is the engine that makes agent memory sustainable at scale.
    """

    def __init__(
        self,
        memory,  # Memory instance
        supermemory_bridge,  # SupermemoryBridge instance
        ollama_client,  # OllamaClient instance
        ledger: TokenLedger,
        scorer_model: str = "llama3.1:8b-instruct-q8_0",
        quality_gate: int = 3,
        project_excluded_sources: Optional[set] = None,
    ):
        self._memory = memory
        self._bridge = supermemory_bridge
        self._scorer = MemoryScorer(
            ollama_client, model=scorer_model, min_score=quality_gate
        )
        self._ledger = ledger
        self._project_excluded = (
            project_excluded_sources or DEFAULT_PROJECT_EXCLUDED_SOURCES
        )
        self._sync_state_path = os.path.join(
            os.path.dirname(ledger._path), "sync_state.json"
        )
        self._sync_state = self._load_sync_state()
        self._consecutive_errors = 0
        self._circuit_breaker_limit = 5

    def _load_sync_state(self) -> Dict[str, Any]:
        """Track which memories have been synced (by content hash)."""
        if os.path.exists(self._sync_state_path):
            try:
                with open(self._sync_state_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"synced_hashes": {}, "last_sync": None}

    def _save_sync_state(self):
        with open(self._sync_state_path, "w") as f:
            json.dump(self._sync_state, f, indent=2, default=str)

    def _content_hash(self, fact: str, category: str) -> str:
        """Deterministic hash matching the bridge's dedup logic."""
        import hashlib

        return hashlib.sha256(fact.encode()).hexdigest()[:16]

    def _is_already_synced(self, fact: str, category: str) -> bool:
        h = self._content_hash(fact, category)
        return h in self._sync_state.get("synced_hashes", {})

    def _mark_synced(self, fact: str, category: str):
        h = self._content_hash(fact, category)
        self._sync_state.setdefault("synced_hashes", {})[h] = {
            "category": category,
            "synced_at": datetime.utcnow().isoformat(),
            "preview": fact[:80],
        }

    async def run(self, include_tier2: bool = False) -> SyncReport:
        """
        Execute a metered sync run.

        Args:
            include_tier2: If True, also sync Tier 2 categories (weekly mode).
        """
        report = SyncReport(
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            mode="weekly" if include_tier2 else "daily",
        )

        # Check bridge availability
        if not self._bridge.available:
            report.errors.append("Supermemory bridge unavailable")
            logger.error("Metered sync aborted: bridge unavailable")
            return report

        # Pre-flight budget check
        if not self._ledger.can_spend(TOKENS_PER_MEMORY):
            report.errors.append(
                f"Budget exhausted (daily: {self._ledger.daily_remaining()}, "
                f"monthly: {self._ledger.monthly_remaining()})"
            )
            logger.warning("Metered sync aborted: token budget exhausted")
            return report

        # Phase 1: Decay
        logger.info("Phase 1: Applying decay policies...")
        report.decayed = apply_decay(self._memory, self._memory._data_dir)

        # Phase 2: Tier 1 sync
        logger.info(
            "Phase 2: Syncing Tier 1 (decisions, patterns, lessons, sessions)..."
        )
        scores = []
        for category in TIER_1_CATEGORIES:
            items = self._memory._read_category(category)
            for item in items:
                fact = item.get("fact", "")
                if not fact:
                    continue

                # Skip already-synced memories
                if self._is_already_synced(fact, category):
                    report.tier1_already_synced += 1
                    continue

                # Budget gate
                if not self._ledger.can_spend(TOKENS_PER_MEMORY):
                    report.errors.append("Daily budget exhausted mid-sync")
                    break

                # Circuit breaker
                if self._consecutive_errors >= self._circuit_breaker_limit:
                    report.circuit_breaker_tripped = True
                    report.errors.append(
                        f"Circuit breaker: {self._consecutive_errors} consecutive errors"
                    )
                    logger.error("Circuit breaker tripped — halting sync")
                    break

                # Quality gate
                score_result = await self._scorer.score(fact, category)
                scores.append(score_result["score"])
                report.memories_scored += 1

                if not score_result["worthy"]:
                    report.tier1_skipped += 1
                    logger.debug(
                        f"Skipped [{category}] score={score_result['score']}: "
                        f"{fact[:60]}... — {score_result['reason']}"
                    )
                    continue

                # Sync to cloud
                result = await self._bridge.sync_memory(
                    fact=fact,
                    category=category,
                    source=item.get("source", "metered_sync"),
                    metadata={
                        **item.get("metadata", {}),
                        "quality_score": score_result["score"],
                        "sync_mode": "metered",
                    },
                )

                if result.get("synced"):
                    report.tier1_synced += 1
                    report.tokens_used += TOKENS_PER_MEMORY
                    self._ledger.record_spend(TOKENS_PER_MEMORY, synced=1)
                    self._mark_synced(fact, category)
                    self._consecutive_errors = 0
                else:
                    self._consecutive_errors += 1
                    reason = result.get("reason", "unknown")
                    report.errors.append(f"Sync failed [{category}]: {reason}")

            # Check circuit breaker between categories
            if report.circuit_breaker_tripped:
                break

        # Phase 3: Tier 2 sync (weekly only)
        if include_tier2 and not report.circuit_breaker_tripped:
            logger.info("Phase 3: Syncing Tier 2 (team, agents, meta)...")
            for category in TIER_2_CATEGORIES:
                items = self._memory._read_category(category)
                for item in items:
                    fact = item.get("fact", "")
                    if not fact or self._is_already_synced(fact, category):
                        continue

                    # Skip noisy project sources (health snapshots, code scans)
                    if category == "project":
                        source = item.get("source", "")
                        if source in self._project_excluded:
                            report.tier2_skipped += 1
                            continue

                    if not self._ledger.can_spend(TOKENS_PER_MEMORY):
                        break

                    # Tier 2 skips quality scoring — low volume, all synced
                    result = await self._bridge.sync_memory(
                        fact=fact,
                        category=category,
                        source=item.get("source", "metered_sync"),
                        metadata={
                            **item.get("metadata", {}),
                            "sync_mode": "metered_tier2",
                        },
                    )

                    if result.get("synced"):
                        report.tier2_synced += 1
                        report.tokens_used += TOKENS_PER_MEMORY
                        self._ledger.record_spend(TOKENS_PER_MEMORY, synced=1)
                        self._mark_synced(fact, category)
                    else:
                        report.tier2_skipped += 1

        # Finalize
        if scores:
            report.avg_score = sum(scores) / len(scores)

        report.tokens_remaining_daily = self._ledger.daily_remaining()
        report.tokens_remaining_monthly = self._ledger.monthly_remaining()

        # Record skipped/decayed totals
        total_decayed = sum(report.decayed.values())
        if total_decayed:
            self._ledger.record_spend(0, decayed=total_decayed)

        # Persist sync state
        self._sync_state["last_sync"] = report.timestamp
        self._save_sync_state()

        # Cleanup old ledger entries
        self._ledger.cleanup_old_entries()

        logger.info(
            f"Metered sync complete: {report.tier1_synced + report.tier2_synced} synced, "
            f"{report.tier1_skipped + report.tier2_skipped} skipped, "
            f"{report.tokens_used} tokens used"
        )

        return report
