"""
AIIA Phase 2 Autonomy Module

Extends the execution engine with proactive capabilities:
- ProactiveStoryExecutor: Auto-decomposes and queues P0/P1 stories outside business hours
- GatedDowngradePolicy: Auto-promotes stale low-severity GATED actions to SUPERVISED
- SelfHealingMonitor: Detects unhealthy services, attempts known fix patterns
- MemoryQualityLoop: Local memory consolidation, dedup, and ChromaDB promotion

All features gated behind AutonomyConfig flags. Ship disabled, enable incrementally.
Master switch: AIIA_AUTONOMY_LEVEL=phase2 (env var)
"""

from local_brain.autonomy.gated_downgrade import GatedDowngradePolicy
from local_brain.autonomy.memory_quality import MemoryQualityLoop
from local_brain.autonomy.proactive_executor import ProactiveStoryExecutor
from local_brain.autonomy.self_healing import SelfHealingMonitor

__all__ = [
    "GatedDowngradePolicy",
    "MemoryQualityLoop",
    "ProactiveStoryExecutor",
    "SelfHealingMonitor",
]
