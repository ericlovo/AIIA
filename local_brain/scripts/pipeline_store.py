"""
Pipeline store — JSON-backed CRUD for sales deals.

Storage: ~/.aiia/eq_data/pipeline/deals.json

Usage:
    from local_brain.scripts.pipeline_store import PipelineStore
    store = PipelineStore()
    store.create(company="Acme Corp", contact="Jane Doe", stage="lead", value=75000, product="my-app")
    deals = store.list(stage="qualified")
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path.home() / ".aiia" / "eq_data" / "pipeline"
DEALS_FILE = DATA_DIR / "deals.json"

VALID_STAGES = (
    "lead",
    "qualified",
    "proposal",
    "negotiation",
    "closed_won",
    "closed_lost",
)


class PipelineStore:
    def __init__(self, data_dir: Optional[str] = None):
        self._dir = Path(data_dir) if data_dir else DATA_DIR
        self._file = self._dir / "deals.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._deals: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        self._file.write_text(json.dumps(self._deals, indent=2))

    def list(self, stage: Optional[str] = None) -> List[Dict]:
        results = self._deals
        if stage:
            results = [d for d in results if d.get("stage") == stage]
        return results

    def get(self, deal_id: str) -> Optional[Dict]:
        for d in self._deals:
            if d["id"] == deal_id:
                return d
        return None

    def create(
        self,
        company: str,
        contact: str = "",
        stage: str = "lead",
        value: float = 0,
        product: str = "",
        notes: str = "",
    ) -> Dict:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Use {VALID_STAGES}")

        deal = {
            "id": uuid.uuid4().hex[:8],
            "company": company,
            "contact": contact,
            "stage": stage,
            "value": value,
            "product": product,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._deals.append(deal)
        self._save()
        return deal

    def update(self, deal_id: str, **fields) -> Optional[Dict]:
        for d in self._deals:
            if d["id"] == deal_id:
                if "stage" in fields and fields["stage"] not in VALID_STAGES:
                    raise ValueError(f"Invalid stage: {fields['stage']}")
                for key in ("company", "contact", "stage", "value", "product", "notes"):
                    if key in fields:
                        d[key] = fields[key]
                d["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return d
        return None

    def delete(self, deal_id: str) -> bool:
        before = len(self._deals)
        self._deals = [d for d in self._deals if d["id"] != deal_id]
        if len(self._deals) < before:
            self._save()
            return True
        return False

    def summary(self) -> Dict:
        """Return funnel summary with counts and values per stage."""
        by_stage = {}
        for stage in VALID_STAGES:
            deals = [d for d in self._deals if d.get("stage") == stage]
            by_stage[stage] = {
                "count": len(deals),
                "value": sum(d.get("value", 0) for d in deals),
            }
        return {
            "total_deals": len(self._deals),
            "total_value": sum(d.get("value", 0) for d in self._deals),
            "by_stage": by_stage,
        }
