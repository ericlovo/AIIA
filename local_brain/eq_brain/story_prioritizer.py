"""
Story Prioritizer — LLM-scored prioritization for roadmap stories.

Scores stories against the 5-filter priority framework:
  1. Does it close a deal?
  2. Does it retain the primary client?
  3. Does it reduce cost?
  4. Does it enable multiple tenants?
  5. Does it create a new revenue stream?

Also extracts candidate stories from session end data (next_steps, blockers).
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.story_prioritizer")

# ---------------------------------------------------------------------------
# Geometric scoring — vector-based priority analysis
#
# Instead of just summing weighted intervals, we also treat each story as a
# vector in 5D filter space and measure its alignment with an "ideal" business
# impact direction. This captures non-linear interactions: a story that moves
# TWO high-weight needles is more valuable than the additive sum suggests.
#
# Composite score = 70% additive (interpretable) + 30% geometric (interaction-aware)
# ---------------------------------------------------------------------------

# Ideal business impact direction — same weights as filters, normalized
_IDEAL_RAW = [5.0, 4.0, 3.0, 2.0, 1.0]
_IDEAL_MAG = math.sqrt(sum(x * x for x in _IDEAL_RAW))
_IDEAL_UNIT = [x / _IDEAL_MAG for x in _IDEAL_RAW]
# Max possible geometric score (all filters at 10, perfect alignment)
_MAX_GEO = math.sqrt(sum(10.0**2 for _ in _IDEAL_RAW))  # ~22.36

ADDITIVE_WEIGHT = 0.7
GEOMETRIC_WEIGHT = 0.3


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def geometric_score(filter_scores: Dict[str, int]) -> Dict[str, float]:
    """Compute vector-based priority metrics from filter scores.

    Returns alignment (cosine similarity to ideal), magnitude, and
    a geometric score that rewards stories moving multiple high-weight needles.
    """
    vec = [
        float(filter_scores.get("closes_deal", 0)),
        float(filter_scores.get("retains_client", 0)),
        float(filter_scores.get("reduces_cost", 0)),
        float(filter_scores.get("enables_tenants", 0)),
        float(filter_scores.get("new_revenue", 0)),
    ]

    mag = _magnitude(vec)
    if mag == 0:
        return {"alignment": 0.0, "magnitude": 0.0, "geometric_score": 0.0}

    unit = [x / mag for x in vec]
    alignment = _dot(unit, _IDEAL_UNIT)
    geo = alignment * mag

    return {
        "alignment": round(alignment, 3),
        "magnitude": round(mag, 2),
        "geometric_score": round(geo, 2),
    }


def composite_score(weighted_total: float, geo: Dict[str, float]) -> float:
    """Blend additive and geometric scores into a single 0-100 composite.

    70% additive (interpretable, max 150) + 30% geometric (interaction-aware).
    """
    additive_norm = weighted_total / 150.0  # 0-1
    geo_norm = geo.get("geometric_score", 0.0) / _MAX_GEO  # 0-1
    raw = ADDITIVE_WEIGHT * additive_norm + GEOMETRIC_WEIGHT * geo_norm
    return round(raw * 100, 1)


# Priority framework — scored 0-10 per filter, weighted
PRIORITY_FILTERS = [
    {
        "name": "closes_deal",
        "weight": 5,
        "question": "Does this help close an active deal?",
    },
    {
        "name": "retains_client",
        "weight": 4,
        "question": "Does this fix a bug, improve UX, or add a feature that retains the primary client?",
    },
    {
        "name": "reduces_cost",
        "weight": 3,
        "question": "Does this reduce operational cost (token spend, infra, manual work)?",
    },
    {
        "name": "enables_tenants",
        "weight": 2,
        "question": "Does this enable multiple tenants or improve the platform for all products?",
    },
    {
        "name": "new_revenue",
        "weight": 1,
        "question": "Does this create a new revenue stream (new product)?",
    },
]

PRIORITIZE_SYSTEM = """\
You are a product prioritization engine for a multi-tenant AI platform for professional services.

Score each story against these filters (0-10 each):
1. Closes a deal? (weight 5) — active pipeline deals
2. Retains primary client? (weight 4) — bug fix, UX improvement, feature for the paying client
3. Reduces cost? (weight 3) — token spend, infra, manual overhead
4. Enables multiple tenants? (weight 2) — platform capability, not one-off customization
5. New revenue stream? (weight 1) — new product line or GA release

Respond with ONLY a JSON object:
{
  "scores": {"closes_deal": N, "retains_client": N, "reduces_cost": N, "enables_tenants": N, "new_revenue": N},
  "weighted_total": N,
  "suggested_priority": "P0|P1|P2|P3",
  "reasoning": "one sentence why"
}"""

PRIORITIZE_PROMPT = """\
Score this story:

Title: {title}
Product: {product}
Current Priority: {priority}
Description: {description}
Tags: {tags}
Client Impact: {client_impact}"""

EXTRACT_STORIES_SYSTEM = """\
You are a product backlog extractor for a multi-tenant AI platform.

Given session end data (summary, next steps, blockers), extract concrete stories that should be tracked.

Rules:
- Only extract items that represent real work (features, bugs, tech debt, integrations)
- Skip vague items like "continue working on X" or "think about Y"
- Each story needs a clear, actionable title
- Assign a product: platform, local-brain, or the most relevant product name
- Tag with: feature, bug, tech-debt, integration, ux, security, performance, or devops

Respond with ONLY a JSON array (empty array if nothing to extract):
[
  {
    "title": "Short actionable title",
    "product": "product-name",
    "description": "What needs to happen",
    "tags": ["feature"],
    "client_impact": "who benefits and why (or empty)"
  }
]"""

EXTRACT_STORIES_PROMPT = """\
Extract backlog stories from this session:

Summary: {summary}
Key Decisions: {decisions}
Next Steps: {next_steps}
Blockers: {blockers}"""


class StoryPrioritizer:
    def __init__(self, ollama_client=None, model: str = ""):
        self._ollama = ollama_client
        self._model = model or "llama3.1:8b-instruct-q8_0"

    async def score_story(self, story: Dict) -> Dict[str, Any]:
        """Score a single story against the priority framework."""
        if not self._ollama:
            return {"error": "No LLM available for scoring"}

        prompt = PRIORITIZE_PROMPT.format(
            title=story.get("title", ""),
            product=story.get("product", ""),
            priority=story.get("priority", "P2"),
            description=story.get("description", ""),
            tags=", ".join(story.get("tags", [])),
            client_impact=story.get("client_impact", ""),
        )

        try:
            response = await self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                system=PRIORITIZE_SYSTEM,
                temperature=0.2,
                max_tokens=512,
                num_ctx=4096,
            )
        except Exception as e:
            logger.error(f"Story scoring failed: {e}")
            return {"error": str(e)}

        content = response.get("message", {}).get("content", "")
        return self._parse_score(content)

    async def prioritize_backlog(
        self, stories: List[Dict], limit: int = 10
    ) -> List[Dict]:
        """Score and rank a list of backlog stories. Returns sorted by weighted score."""
        scorable = [s for s in stories if s.get("status") in ("backlog", "active")][
            :limit
        ]

        results = []
        for story in scorable:
            score = await self.score_story(story)
            if "error" not in score:
                results.append(
                    {
                        **story,
                        "priority_score": score.get("weighted_total", 0),
                        "composite_score": score.get("composite_score", 0.0),
                        "geometric": score.get("geometric", {}),
                        "priority_reasoning": score.get("reasoning", ""),
                        "suggested_priority": score.get("suggested_priority", "P2"),
                        "filter_scores": score.get("scores", {}),
                    }
                )
            else:
                results.append(
                    {**story, "priority_score": 0, "composite_score": 0.0, "score_error": score["error"]}
                )

        results.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
        return results

    async def extract_stories_from_session(
        self,
        summary: str,
        next_steps: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        key_decisions: Optional[List[str]] = None,
        session_id: str = "",
    ) -> List[Dict]:
        """Extract candidate stories from session end data."""
        if not self._ollama:
            return []

        # Don't extract if there's nothing to work with
        steps_text = "; ".join(next_steps) if next_steps else ""
        blockers_text = "; ".join(blockers) if blockers else ""
        decisions_text = "; ".join(key_decisions) if key_decisions else ""

        if not steps_text and not blockers_text:
            return []

        prompt = EXTRACT_STORIES_PROMPT.format(
            summary=summary,
            decisions=decisions_text or "none",
            next_steps=steps_text or "none",
            blockers=blockers_text or "none",
        )

        try:
            response = await self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                system=EXTRACT_STORIES_SYSTEM,
                temperature=0.3,
                max_tokens=1024,
                num_ctx=4096,
            )
        except Exception as e:
            logger.error(f"Story extraction failed: {e}")
            return []

        content = response.get("message", {}).get("content", "")
        candidates = self._parse_candidates(content)

        # Tag with session source
        for c in candidates:
            c["source_session"] = session_id
            c["source_type"] = "auto-extracted"

        return candidates

    def _parse_score(self, content: str) -> Dict:
        """Parse LLM scoring output."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                try:
                    result = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    return {"error": f"Could not parse score: {content[:200]}"}
            else:
                return {"error": f"No JSON found in score: {content[:200]}"}

        if not isinstance(result, dict):
            return {"error": "Score is not a dict"}

        # Always calculate weighted total from filter scores (don't trust LLM math)
        if "scores" in result:
            scores = result["scores"]
            # Clamp each score to 0-10
            for key in scores:
                scores[key] = max(0, min(10, scores.get(key, 0)))
            total = sum(
                scores.get(f["name"], 0) * f["weight"] for f in PRIORITY_FILTERS
            )
            result["weighted_total"] = total

            # Geometric scoring — vector analysis of filter scores
            geo = geometric_score(scores)
            result["geometric"] = geo
            result["composite_score"] = composite_score(total, geo)
        elif "weighted_total" not in result:
            result["weighted_total"] = 0
            result["composite_score"] = 0.0

        # Priority from composite score (0-100 scale)
        cs = result.get("composite_score", 0.0)
        if cs >= 55:
            result["suggested_priority"] = "P0"
        elif cs >= 35:
            result["suggested_priority"] = "P1"
        elif cs >= 18:
            result["suggested_priority"] = "P2"
        else:
            result["suggested_priority"] = "P3"

        return result

    def _parse_candidates(self, content: str) -> List[Dict]:
        """Parse LLM candidate story output."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    content = part
                    break

        try:
            candidates = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end > start:
                try:
                    candidates = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse candidates: {content[:200]}")
                    return []
            else:
                return []

        if not isinstance(candidates, list):
            return []

        validated = []
        for c in candidates:
            if not isinstance(c, dict) or not c.get("title"):
                continue
            validated.append(
                {
                    "title": c["title"],
                    "product": c.get("product", "platform"),
                    "description": c.get("description", ""),
                    "tags": c.get("tags", []),
                    "client_impact": c.get("client_impact", ""),
                }
            )

        return validated
