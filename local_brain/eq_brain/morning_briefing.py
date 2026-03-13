"""
Morning Briefing — Smart alerting + code review synthesis.

Runs nightly at 4:30am using DeepSeek R1 (14b) to:
- Read overnight reports (security, sync, daily code)
- Pre-analyze commits for risky patterns (pure heuristics, no LLM)
- Feed everything to DeepSeek for severity-ranked synthesis
- Create ActionQueue items for critical/error findings
- Store briefing summary as meta memory

Output: JSON with alerts (severity-ranked), summary, code_review_notes, health_grade (A-F)
Fallback: if DeepSeek fails, build briefing from pre-analyzed risk flags only.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_brain.eq_brain.brain import AIIA
from local_brain.eq_brain.memory import Memory

logger = logging.getLogger("aiia.briefing")

# Risky file path patterns — pre-LLM heuristic classification
RISKY_PATTERNS: Dict[str, Dict[str, Any]] = {
    "auth": {
        "paths": ["auth.py", "auth/", "jwt", "login", "password", "token"],
        "severity": "critical",
    },
    "dependencies": {
        "paths": ["requirements.txt", "package.json", "Pipfile", "poetry.lock"],
        "severity": "warn",
    },
    "database": {
        "paths": ["migration", "models/", "alembic", ".sql"],
        "severity": "error",
    },
    "secrets": {
        "paths": [".env", "credentials", "secret", "api_key"],
        "severity": "critical",
    },
    "security": {
        "paths": ["security/", "middleware/", "cors", "rate_limit", "encryption"],
        "severity": "error",
    },
    "billing": {
        "paths": ["billing", "payment", "stripe", "subscription", "pricing"],
        "severity": "error",
    },
}

# Map alert categories to ActionQueue types
ALERT_TO_ACTION_TYPE = {
    "security": "security_fix",
    "auth": "security_fix",
    "secrets": "security_fix",
    "database": "review",
    "dependencies": "review",
    "billing": "review",
    "code_quality": "post_commit_review",
    "tech_debt": "tech_debt",
}

BRIEFING_PROMPT = """You are AIIA's morning briefing engine. Analyze overnight reports and produce an executive intelligence briefing for Eric.

## Security Scan Report
{security_report}

## Memory Sync Report
{sync_report}

## Daily Code Report
{code_report}

## Pre-Analyzed Risk Flags
{risk_flags}

## Baseline Context
The security scan has a `.security-baseline.json` file. Findings that match the baseline are ACCEPTED and should NOT be treated as failures. Look for "effective: PASS" or "baselined" in the Overall line — if present, the scan is effectively passing despite individual scanner FAIL/WARN markers on baselined findings. Only flag genuinely NEW findings not covered by the baseline.

## Instructions

Produce a JSON object with these fields:

1. **alerts**: Array of findings ranked by severity. Each has:
   - "severity": "critical" | "error" | "warn" | "info"
   - "category": string (e.g., "security", "code_quality", "dependencies")
   - "title": short description (under 80 chars)
   - "detail": 1-2 sentence explanation
   - "files": array of affected file paths (if applicable)

2. **summary**: 2-3 sentence executive summary of overnight activity.

3. **code_review_notes**: Array of notable observations about yesterday's commits (patterns, concerns, praise). Each has "note" and "severity" ("info" | "warn").

4. **health_grade**: Single letter A-F. A=everything clean, B=minor issues, C=needs attention, D=significant concerns, F=critical issues.

Prioritize actionable findings. Skip noise. Return ONLY valid JSON."""


class MorningBriefing:
    """Reads overnight reports, synthesizes with DeepSeek, creates action items."""

    def __init__(
        self,
        aiia: AIIA,
        memory: Memory,
        action_queue: Any = None,
    ):
        self._aiia = aiia
        self._memory = memory
        self._action_queue = action_queue
        self._logs_dir = Path.home() / ".aiia" / "logs"
        self._reports_dir = Path.home() / ".aiia" / "eq_data" / "reports"

    def _read_security_report(self) -> str:
        """Read latest security scan summary + scanner details."""
        summary_path = self._logs_dir / "security" / "latest.txt"
        if not summary_path.exists():
            return "No security scan report found."

        summary = summary_path.read_text().strip()

        # Read individual scanner JSONs for detail
        report_dir = self._logs_dir / "security" / "latest-report"
        scanner_summaries = []
        if report_dir.exists() and report_dir.is_dir():
            for scanner_file in sorted(report_dir.glob("*.json")):
                scanner_name = scanner_file.stem
                try:
                    data = json.loads(scanner_file.read_text())
                    scanner_summaries.append(
                        self._summarize_scanner(scanner_name, data)
                    )
                except (json.JSONDecodeError, OSError) as e:
                    scanner_summaries.append(f"{scanner_name}: error reading ({e})")

        if scanner_summaries:
            summary += "\n\nScanner Details:\n" + "\n".join(scanner_summaries)
        return summary

    def _summarize_scanner(self, name: str, data: Any) -> str:
        """Summarize a single scanner's JSON output."""
        if name == "bandit":
            if isinstance(data, dict):
                results = data.get("results", [])
                if not results:
                    return "bandit: PASS (0 findings)"
                by_sev = {}
                for r in results:
                    sev = r.get("issue_severity", "UNKNOWN")
                    by_sev[sev] = by_sev.get(sev, 0) + 1
                return f"bandit: {len(results)} findings ({by_sev})"
            return f"bandit: {len(data) if isinstance(data, list) else '?'} findings"

        if name == "semgrep":
            if isinstance(data, dict):
                results = data.get("results", [])
                errors = data.get("errors", [])
                if not results:
                    return f"semgrep: PASS (0 findings, {len(errors)} errors)"
                return f"semgrep: {len(results)} findings, {len(errors)} errors"
            return "semgrep: unknown format"

        if name == "trivy-deps":
            if isinstance(data, dict):
                results = data.get("Results", [])
                total_vulns = sum(
                    len(r.get("Vulnerabilities", []))
                    for r in results
                    if isinstance(r, dict)
                )
                return f"trivy: {total_vulns} vulnerabilities"
            return "trivy: unknown format"

        if name == "trufflehog":
            if isinstance(data, list):
                return (
                    f"trufflehog: {len(data)} findings" if data else "trufflehog: PASS"
                )
            return "trufflehog: unknown format"

        if name == "shellcheck":
            if isinstance(data, list):
                return (
                    f"shellcheck: {len(data)} findings" if data else "shellcheck: PASS"
                )
            return "shellcheck: unknown format"

        if name == "hadolint":
            if isinstance(data, list):
                return f"hadolint: {len(data)} findings" if data else "hadolint: PASS"
            return "hadolint: unknown format"

        return f"{name}: present"

    def _read_sync_report(self) -> str:
        """Read latest memory sync summary."""
        summary_path = self._logs_dir / "sync" / "latest.txt"
        if not summary_path.exists():
            return "No memory sync report found."
        return summary_path.read_text().strip()

    def _read_daily_report(self) -> str:
        """Read latest daily code report."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        # Try today first, then yesterday
        for date_str in [today, self._yesterday()]:
            report_path = self._reports_dir / f"{date_str}.json"
            if report_path.exists():
                try:
                    data = json.loads(report_path.read_text())
                    return self._format_code_report(data)
                except (json.JSONDecodeError, OSError):
                    continue
        return "No daily code report found."

    def _yesterday(self) -> str:
        from datetime import timedelta

        return (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    def _format_code_report(self, data: Dict[str, Any]) -> str:
        """Format the daily code report JSON into a readable string."""
        summary = data.get("summary", {})
        lines = [
            f"Date: {data.get('date', '?')}",
            f"Commits: {summary.get('total_commits', 0)}",
            f"Files changed: {summary.get('total_files_changed', 0)}",
            f"Additions: {summary.get('total_additions', 0)}",
            f"Deletions: {summary.get('total_deletions', 0)}",
            f"Authors: {', '.join(summary.get('authors', []))}",
            f"Commit types: {summary.get('commit_types', {})}",
        ]

        products = data.get("products", {})
        for pname, pdata in products.items():
            commits = pdata.get("commits", [])
            if commits:
                lines.append(f"\n{pname}:")
                for c in commits[:10]:  # Cap at 10 per product
                    files = c.get("files", [])
                    file_list = ", ".join(
                        (f if isinstance(f, str) else f.get("path", "?"))
                        for f in files[:5]
                    )
                    lines.append(f"  [{c.get('type', '?')}] {c.get('subject', '?')}")
                    if file_list:
                        lines.append(f"    Files: {file_list}")

        syntax = data.get("syntax_errors", {})
        if isinstance(syntax, dict):
            total_errs = syntax.get("total_errors", 0)
            if total_errs:
                lines.append(f"\nSyntax errors: {total_errs}")
                for product, errs in syntax.get("by_product", {}).items():
                    lines.append(f"  {product}: {errs}")
        elif isinstance(syntax, list) and syntax:
            lines.append(f"\nSyntax errors: {len(syntax)}")
            for err in syntax[:5]:
                lines.append(f"  - {err}")

        return "\n".join(lines)

    def _classify_commit_risk(self, commit: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pre-LLM heuristic: scan commit's changed files for risky patterns."""
        flags = []
        files = commit.get("files", [])
        subject = commit.get("subject", "").lower()

        for f in files:
            file_path = (f if isinstance(f, str) else f.get("path", "")).lower()
            for category, config in RISKY_PATTERNS.items():
                for pattern in config["paths"]:
                    if pattern in file_path:
                        flags.append(
                            {
                                "category": category,
                                "severity": config["severity"],
                                "file": f if isinstance(f, str) else f.get("path", "?"),
                                "commit": commit.get("hash", "?"),
                                "subject": commit.get("subject", "?"),
                            }
                        )
                        break  # One flag per category per file

        return flags

    def _gather_risk_flags(self, code_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan all commits for risky patterns."""
        all_flags = []
        for pname, pdata in code_report.get("products", {}).items():
            for commit in pdata.get("commits", []):
                flags = self._classify_commit_risk(commit)
                all_flags.extend(flags)
        return all_flags

    async def generate(self) -> Dict[str, Any]:
        """Generate the morning briefing.

        Phase 1: Gather overnight reports
        Phase 2: Pre-analyze commits for risky patterns
        Phase 3: Feed to DeepSeek for synthesis
        Phase 4: Create ActionQueue items for critical/error findings
        Phase 5: Store briefing as meta memory
        """
        start = time.monotonic()

        # Phase 1: Gather reports
        security_report = self._read_security_report()
        sync_report = self._read_sync_report()

        # Load raw code report for risk analysis
        code_report_data = {}
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for date_str in [today, self._yesterday()]:
            report_path = self._reports_dir / f"{date_str}.json"
            if report_path.exists():
                try:
                    code_report_data = json.loads(report_path.read_text())
                    break
                except (json.JSONDecodeError, OSError):
                    continue

        code_report = (
            self._format_code_report(code_report_data)
            if code_report_data
            else "No daily code report found."
        )

        # Phase 2: Pre-analyze risk
        risk_flags = self._gather_risk_flags(code_report_data)
        risk_flags_text = "None" if not risk_flags else json.dumps(risk_flags, indent=2)
        logger.info(f"Pre-analysis: {len(risk_flags)} risk flags found")

        # Phase 3: Deep synthesis
        briefing = await self._deep_synthesis(
            security_report, sync_report, code_report, risk_flags_text
        )

        # Fallback: if DeepSeek failed, build from heuristics
        if not briefing:
            logger.warning("DeepSeek synthesis failed — using heuristic fallback")
            briefing = self._heuristic_fallback(
                security_report, sync_report, code_report_data, risk_flags
            )

        # Phase 4: Create ActionQueue items
        actions_created = 0
        if self._action_queue and briefing.get("alerts"):
            actions_created = self._create_actions(briefing["alerts"])

        # Phase 5: Store as meta memory
        grade = briefing.get("health_grade", "?")
        alert_count = len(briefing.get("alerts", []))
        summary = briefing.get("summary", "No summary")
        self._memory.remember(
            fact=f"Morning briefing ({today}): Grade {grade}, {alert_count} alerts. {summary}",
            category="meta",
            source="briefing",
            metadata={
                "type": "morning_briefing",
                "health_grade": grade,
                "alert_count": alert_count,
                "actions_created": actions_created,
            },
        )

        elapsed = (time.monotonic() - start) * 1000
        briefing["latency_ms"] = round(elapsed, 1)
        briefing["timestamp"] = datetime.utcnow().isoformat()
        briefing["risk_flags_count"] = len(risk_flags)
        briefing["actions_created"] = actions_created

        logger.info(
            f"Briefing complete: grade={grade}, {alert_count} alerts, "
            f"{actions_created} actions in {elapsed:.0f}ms"
        )
        return briefing

    async def _deep_synthesis(
        self,
        security_report: str,
        sync_report: str,
        code_report: str,
        risk_flags: str,
    ) -> Optional[Dict[str, Any]]:
        """Feed reports to DeepSeek for intelligent synthesis."""
        prompt = BRIEFING_PROMPT.format(
            security_report=security_report,
            sync_report=sync_report,
            code_report=code_report,
            risk_flags=risk_flags,
        )

        try:
            result = await self._aiia.ask(
                question=prompt,
                include_sessions=False,
                n_results=0,
                depth="deep",
            )

            raw_answer = result.get("answer", "")
            reasoning = result.get("reasoning", "")
            if reasoning:
                logger.debug(f"DeepSeek briefing reasoning ({len(reasoning)} chars)")

            # Reuse consolidator's JSON parser
            from local_brain.eq_brain.memory_consolidator import (
                MemoryConsolidator,
            )

            parsed = MemoryConsolidator._parse_json_response(raw_answer)
            if parsed:
                return parsed

            logger.warning("Could not parse DeepSeek briefing output")
            return None

        except Exception as e:
            logger.error(f"DeepSeek synthesis failed: {e}", exc_info=True)
            return None

    def _heuristic_fallback(
        self,
        security_report: str,
        sync_report: str,
        code_report_data: Dict[str, Any],
        risk_flags: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a basic briefing from heuristics when DeepSeek is unavailable."""
        alerts = []

        # Promote risk flags to alerts
        for flag in risk_flags:
            alerts.append(
                {
                    "severity": flag["severity"],
                    "category": flag["category"],
                    "title": f"Risky change in {flag['file']}",
                    "detail": f"Commit {flag['commit']}: {flag['subject']}",
                    "files": [flag["file"]],
                }
            )

        # Check security report for failures (respect baseline)
        has_fail = "FAIL" in security_report
        baseline_pass = "effective: PASS" in security_report
        if has_fail and not baseline_pass:
            alerts.append(
                {
                    "severity": "error",
                    "category": "security",
                    "title": "Security scan has NEW failures",
                    "detail": "One or more security scanners reported findings not covered by baseline. Check logs/security/latest.txt",
                    "files": [],
                }
            )

        # Check sync report
        if "FAIL" in sync_report:
            alerts.append(
                {
                    "severity": "warn",
                    "category": "sync",
                    "title": "Memory sync had errors",
                    "detail": "Nightly memory sync reported errors. Check logs/sync/latest.txt",
                    "files": [],
                }
            )

        # Determine grade from alerts
        severity_scores = {"critical": 4, "error": 3, "warn": 1, "info": 0}
        total_score = sum(severity_scores.get(a["severity"], 0) for a in alerts)
        if total_score == 0:
            grade = "A"
        elif total_score <= 2:
            grade = "B"
        elif total_score <= 5:
            grade = "C"
        elif total_score <= 10:
            grade = "D"
        else:
            grade = "F"

        summary_parts = []
        commit_count = code_report_data.get("summary", {}).get("total_commits", 0)
        if commit_count:
            summary_parts.append(f"{commit_count} commits shipped overnight.")
        if risk_flags:
            summary_parts.append(f"{len(risk_flags)} risky file changes detected.")
        if not summary_parts:
            summary_parts.append("Quiet night — no significant activity.")

        return {
            "alerts": sorted(
                alerts,
                key=lambda a: severity_scores.get(a["severity"], 0),
                reverse=True,
            ),
            "summary": " ".join(summary_parts),
            "code_review_notes": [],
            "health_grade": grade,
            "fallback": True,
        }

    def _create_actions(self, alerts: List[Dict[str, Any]]) -> int:
        """Create ActionQueue items for critical and error severity alerts.

        Deduplicates against existing pending actions by title to prevent
        the same baselined findings from piling up nightly.
        """
        # Build a set of existing pending action titles for O(1) dedup
        existing_titles = set()
        if self._action_queue:
            for action in self._action_queue.list_actions(status="pending", limit=200):
                existing_titles.add(action.get("title", ""))

        created = 0
        skipped = 0
        for alert in alerts:
            severity = alert.get("severity", "info")
            if severity not in ("critical", "error"):
                continue

            title = alert.get("title", "Briefing alert")
            if title in existing_titles:
                skipped += 1
                continue

            category = alert.get("category", "code_quality")
            action_type = ALERT_TO_ACTION_TYPE.get(category, "review")

            self._action_queue.create_action(
                action_type=action_type,
                severity=severity,
                title=title,
                description=alert.get("detail", ""),
                source_task="morning_briefing",
                files_affected=alert.get("files", []),
            )
            existing_titles.add(title)
            created += 1

        if created or skipped:
            logger.info(
                f"Created {created} ActionQueue items from briefing alerts"
                + (f" ({skipped} duplicates skipped)" if skipped else "")
            )
        return created
