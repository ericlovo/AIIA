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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.story_prioritizer")

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
                        "priority_reasoning": score.get("reasoning", ""),
                        "suggested_priority": score.get("suggested_priority", "P2"),
                        "filter_scores": score.get("scores", {}),
                    }
                )
            else:
                results.append(
                    {**story, "priority_score": 0, "score_error": score["error"]}
                )

        results.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
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
        elif "weighted_total" not in result:
            result["weighted_total"] = 0

        # Always override LLM's priority suggestion with our thresholds
        # Max possible = 150 (all filters score 10)
        # P0 = 90+ (60%+) — truly critical, moves multiple needles
        # P1 = 50-89 (33-59%) — important, clear business impact
        # P2 = 25-49 (17-32%) — normal backlog
        # P3 = 0-24 (<17%) — nice to have
        total = result.get("weighted_total", 0)
        if total >= 90:
            result["suggested_priority"] = "P0"
        elif total >= 50:
            result["suggested_priority"] = "P1"
        elif total >= 25:
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
