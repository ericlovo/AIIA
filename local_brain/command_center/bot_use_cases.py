"""Configurable task shortcuts; defaults never change a bot's execution settings."""

from typing import Any

DEFAULTS = {
    "alumni nations research scout": [
        (
            "Research brief",
            "Prepare a sourced, dated Alumni Nations research brief grounded in project memory. Mark unknowns [CONFIRM]; research only, no outreach.",
        ),
        (
            "Check evidence",
            "Assess the evidence behind Alumni Nations engagement claims. Distinguish university statistics, associations, and unconfirmed claims.",
        ),
        (
            "Find research gaps",
            "Identify the three most useful Alumni Nations research gaps using project memory and the limits of the survey evidence.",
        ),
    ],
    "aiia product lead": [
        (
            "Product status",
            "Report what is shipped versus proposed for AIIA Console, Studio, and Voice, citing repository and GitHub evidence.",
        ),
        (
            "Draft changelog",
            "Draft changelog entries from merged AIIA work, with commit or PR evidence. Propose text only.",
        ),
        (
            "What to build next",
            "Recommend the next three AIIA product improvements with evidence, trade-offs, and a clear first step.",
        ),
    ],
    "ci & failure fixer": [
        (
            "Triage red checks",
            "Inspect current AIIA CI and open PRs. Separate active failures from stale history and name the failing check and ownership area.",
        ),
        (
            "Reproduce failure",
            "Propose the narrowest reproduction command for a confirmed CI failure, with expected versus observed behavior. Do not claim commands were run.",
        ),
        (
            "Propose smallest fix",
            "Propose the smallest fix for a confirmed failure and the verification that would prove it. Read-only; no code changes.",
        ),
    ],
    "failure reproducer": [
        (
            "Bound a failure",
            "Turn a confirmed failure into a bounded reproduction assignment with expected and observed behavior.",
        ),
        (
            "Find a repro command",
            "Identify the narrowest test command and affected surface for a confirmed failure. Do not execute changes.",
        ),
        (
            "Define fix evidence",
            "Specify the evidence needed before fixing a confirmed failure. Propose only; no Git workspace.",
        ),
    ],
    "code reviewer": [
        (
            "Review a PR",
            "Review the supplied PR, diff, or commit range for correctness, security, and compatibility risks. Read-only.",
        ),
        (
            "Find test gaps",
            "Map the supplied change to existing tests, identify the highest-risk gaps, and propose the smallest useful test.",
        ),
        (
            "Review dependencies",
            "Review dependency and CI-action bumps as a set. Group compatibility risks, security urgency, and a safe merge order. Read-only.",
        ),
    ],
    "repo brief & implementer": [
        (
            "Implementation plan",
            "Draft an evidence-backed implementation plan for the supplied goal: files, risks, tests, and rollback. Plan only; no workspace or changes.",
        ),
        (
            "Release readiness",
            "Prepare a GO, HOLD, or NO-GO release checklist from branch state, CI, approved changes, migration risk, and rollback evidence. Do not release.",
        ),
        (
            "Scope an isolated change",
            "Propose a bounded isolated-workspace assignment for the supplied goal, including acceptance criteria. Do not create a workspace or implement without human approval.",
        ),
    ],
    "dependency diplomat": [
        (
            "Upgrade clusters",
            "Group proposed AIIA dependency and CI-action upgrades by compatibility. Read-only.",
        ),
        (
            "Security urgency",
            "Assess security urgency and likely breaking surfaces in proposed dependency upgrades, citing visible evidence.",
        ),
        (
            "Safe merge order",
            "Propose a safe merge order and verification plan for pending dependency upgrades. Do not merge or change code.",
        ),
    ],
    "mindmoor delivery lead": [
        (
            "Delivery risks",
            "Report Mindmoor delivery risks with a one-line verdict, failing CI, stale branches, migrations, and work needing an owner. Cite visible evidence.",
        ),
        (
            "Release branch lag",
            "Compare Mindmoor production and alumni branches with main. Report commit lag and material release risk from visible repository evidence.",
        ),
        (
            "CI and stale branches",
            "Review Mindmoor CI and stale branches for material changes needing an operator decision. Keep a clean day to a short all-clear.",
        ),
    ],
    "sanction product lead": [
        (
            "Review approval contract",
            "Review the supplied Sanction approval contract: actor, action, resource, governing rule, approval state, audit evidence, and denial behavior. Read-only.",
        ),
        (
            "Find fail-open risks",
            "Inspect supplied Sanction policy behavior for fail-open paths and ambiguous ownership. Cite evidence and propose fixes only.",
        ),
        (
            "Policy priorities",
            "Recommend the next Sanction authorization policy improvements with evidence and trade-offs. Read-only.",
        ),
    ],
    "client daily preparer": [
        (
            "Daily evidence pack",
            "Assemble a client daily evidence pack: shipped work, in-flight work, risks, and decisions. Tie each item to a source and date; mark unknowns [CONFIRM].",
        ),
        (
            "Draft daily summary",
            "Draft a short client-ready daily summary from supplied evidence for human review. Mark unknowns [CONFIRM]; do not send anything.",
        ),
        (
            "Onboarding guide",
            "Draft an onboarding path for the named person: goal, ordered stages, concrete outcomes, exact first action, and completion checks. Never invent links or credentials; mark unknowns [CONFIRM].",
        ),
    ],
}


def normalize_use_cases(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError("invalid_use_cases")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "task"}:
            raise ValueError("invalid_use_cases")
        label, task = item["label"], item["task"]
        if not isinstance(label, str) or not isinstance(task, str):
            raise ValueError("invalid_use_cases")
        label, task = label.strip(), task.strip()
        if not label or not task or len(label) > 80 or len(task) > 6000:
            raise ValueError("invalid_use_cases")
        result.append({"label": label, "task": task})
    return result


def resolve_use_cases(agent: dict[str, Any]) -> list[dict[str, str]]:
    if "use_cases" in agent:
        return normalize_use_cases(agent["use_cases"])
    defaults = DEFAULTS.get(str(agent.get("name", "")).strip().lower())
    if defaults:
        return [{"label": label, "task": task} for label, task in defaults]
    description = str(
        agent.get("mission") or agent.get("one_liner") or agent.get("name") or "the supplied goal"
    ).strip()
    return [
        {
            "label": "Brief me",
            "task": f"Using your remit below, prepare a concise evidence-backed brief. Mark missing context and unknowns.\n\n{description}",
        },
        {
            "label": "Find risks",
            "task": f"Using your remit below, identify the three most material risks in the supplied context, with evidence.\n\n{description}",
        },
        {
            "label": "Recommend next steps",
            "task": f"Using your remit below, recommend three prioritized next steps and their trade-offs. Propose only.\n\n{description}",
        },
    ]
