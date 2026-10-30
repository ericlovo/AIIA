"""Conservative routing of captures; classification is not an approval."""

ATTENTION_LANES = ("needs_you", "verification", "activity")
ATTENTION_SQL = """CASE
 WHEN ideas.assignment_id<>'' OR ideas.review_outcome='needs_work' THEN 'needs_you'
 WHEN ideas.attention_lane IN ('verification','activity') THEN ideas.attention_lane
 ELSE 'needs_you' END"""


def classify_idea(idea):
    if idea.get("assignment_id") or idea.get("review_outcome") == "needs_work":
        return (
            "linked_work",
            "Human review or assigned work exists. Review the linked work once; retain this original.",
        )
    if idea.get("author_id") or idea.get("workspace_id"):
        return "needs_you", "Human capture; keep visible until a person decides."
    source = idea.get("source")
    if source == "digest":
        return (
            "activity",
            "Status summary, not a new approval. Underlying decisions remain separate.",
        )
    if source in ("code_review", "backlog_steward", "standup", "public_signals"):
        return (
            "verification",
            "Automated claim or suggestion. Verify current evidence and existing work before escalating.",
        )
    return (
        "needs_you",
        "Unknown or human source. Keep visible; do not infer that it is safe to hide.",
    )


def new_idea_lane(source, author_id="", workspace_id=""):
    return classify_idea(dict(source=source, author_id=author_id, workspace_id=workspace_id))[0]
