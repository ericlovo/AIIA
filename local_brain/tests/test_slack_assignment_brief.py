from local_brain.command_center import attention, daily_digest


def brief(items, agents=()):
    return daily_digest.format_slack_digest(
        date="2026-11-15",
        agents=list(agents),
        assignments=items,
        loops={},
        tasks=[],
        inbox_counts={},
        products=[daily_digest.DigestProduct("platform", "Platform", mount="platform")],
        customers=[daily_digest.DigestCustomer("demo", "Demo client", agents=("Demo Scout",))],
    )


def assignment(**changes):
    return {
        "id": "record",
        "status": "completed",
        "agent_id": "scout",
        "title": "Intake follow-up",
        "result": "A deliverable is ready for review.",
        **changes,
    }


def test_brief_attributes_work_to_its_record_and_links_to_the_exact_assignment():
    scout = {"id": "scout", "name": "Demo Scout", "suite": "platform", "handles": ["demo"]}
    rows = [
        assignment(id="one/1"),
        assignment(id="two", client_id="demo", meeting_id="call"),
        assignment(id="three", client_id="removed-client", agent_id="missing"),
    ]
    text = brief(rows, [scout])
    for scope, record in (
        ("Platform", "one%2F1"),
        ("Demo client", "two"),
        ("Client work", "three"),
    ):
        line = next(line for line in text.splitlines() if "#/assignments/" + record in line)
        assert scope in line and "Intake follow-up" in line


def test_active_check_failure_links_to_inspection_and_successful_retry_uses_latest_result():
    record = assignment(
        status="failed",
        source_kind="loop_check",
        title="Check failed: Delivery watch",
        error="check_incomplete: github_api_timeout, private-diagnostic",
        completed_at="2026-11-14",
    )
    text = brief([record])
    assert "Delivery watch" in text and "GitHub timed out" in text
    assert "Inspect before retrying" in text and "#/assignments/record" in text
    assert "private-diagnostic" not in text and "Review " not in text

    retry = {**record, "status": "completed", "result": "**Verdict:** Clean day", "error": ""}
    assert "#/assignments/record" not in brief([retry])
    review = brief([{**retry, "result": "**Verdict:** Approval needed. Should the watch resume?"}])
    assert "Review the latest check from Delivery watch" in review
    assert "#/assignments/record" in review and "Check failed:" not in review


def test_brief_skips_shelved_records_and_keeps_a_different_call_with_identical_evidence():
    acknowledged = assignment(id="source", client_id="demo", meeting_id="first")
    key = attention.evidence_key(acknowledged)
    items = [
        {**acknowledged, "attention_events": [{"key": key, "active": True}]},
        {**acknowledged, "id": "repeat"},
        {**acknowledged, "id": "different", "meeting_id": "second"},
        assignment(id="dismissed", status="failed", dismissed_at="2026-11-15"),
        assignment(id="enqueued", status="failed", run_ticket=1),
        assignment(id="exercise", title="Continue: Eval AB-12 exercise"),
        assignment(id="active", status="running", attempt_id="run"),
        assignment(id="accepted", review_status="accepted"),
    ]
    text = brief(items)
    assert "#/assignments/different" in text and text.count("#/assignments/") == 1


def test_brief_uses_the_proposal_heading_and_keeps_the_rest_of_the_output_in_studio():
    text = brief(
        [
            assignment(
                title="Retune the watches before scheduling resumes",
                result="```markdown\n## Proposed Configuration for Delivery Watch\n\nprivate body\n```",
            )
        ],
    )
    assert "Review the proposed configuration for Delivery Watch." in text
    assert "before scheduling resumes" not in text
    assert "private body" not in text
    text = brief([assignment(result="## Proposed Configuration for <@everyone>\nprivate body")])
    assert "Intake follow-up" in text and "everyone" not in text and "private body" not in text


def test_brief_shows_recovery_action_for_an_interrupted_save():
    text = brief([assignment(status="failed", attempt_id="run", completed_run_id="")])
    assert "Recover the saved output" in text and "#/assignments/record" in text
    assert "retry" not in text.lower()


def test_specific_collector_blocker_replaces_the_same_agents_generic_pause():
    agent = {"id": "scout", "name": "Source watch", "loop_enabled": False, "loop_task": "collect"}
    other = {**agent, "id": "other", "name": "Other watch"}
    item = assignment(
        title="RFP watch", result="Source collector pending. Begin daily digest delivery to Slack?"
    )
    text = brief([item], [agent, other])
    assert "Verify the RFP source collector" in text and "#/assignments/record" in text
    assert "#/bots/scout" not in text and "#/bots/other" in text


def test_external_schedule_handoff_supersedes_old_local_setup_but_keeps_live_failure():
    watch = {
        "id": "scout",
        "loop_enabled": False,
        "loop_task": "External GitHub Actions owns the daily schedule.",
        "updated_at": "2026-11-10T12:00:00Z",
    }
    proposal = assignment(
        title="RFP watch",
        completed_at="2026-11-01T12:00:00Z",
        result="Source collector pending. Begin daily digest delivery to Slack?",
    )
    current = assignment(
        id="incident", status="failed", title="Check provider", source_kind="loop_check"
    )
    text = brief([proposal, current], [watch])
    assert "#/assignments/record" not in text and "#/bots/scout" not in text
    assert "#/assignments/incident" in text and "Inspect before retrying" in text


def test_counts_proposals_after_filtering_explicit_qa_titles():
    qa_only = (
        "### **PR 1: Exercise the intake page**\n**Title:** `QA ONLY — exercise — DO NOT MERGE`"
    )
    genuine = "### **PR 2: Add coverage for intake**\n**Title:** `test: add coverage`"
    item = assignment(result=qa_only + "\n" + genuine)
    text = brief([item])
    assert "Decide whether to develop" in text and "Add coverage for intake" in text
    assert "Exercise the intake page" not in text
    assert "#/assignments/record" not in brief([{**item, "result": qa_only}])


def test_old_proposals_request_reconciliation_without_presenting_them_as_new_work():
    item = assignment(completed_at="2020-03-15T12:00:00Z", result="### **PR 1: Intake form**")
    text = brief([item])
    assert "Earlier reports awaiting review" in text and "Mar 15 proposal" in text
    assert "#/assignments/record" in text and "*Decisions*" not in text


def test_watch_summary_attributes_intervals_to_the_proposal():
    result = (
        "# Proposed Configuration for Intake and Delivery\n\n## Intake\n- **Cadence**: Daily (08:00 UTC)"
        "\n## Delivery\n- **Cadence**: Every 48 hours (08:00 UTC)\n- **Secret**: private"
    )
    text = brief([assignment(result=result)])
    assert "Review the proposed schedules for Intake (daily) and Delivery (every 48 hours)" in text
    assert "private" not in text


def test_digest_links_saved_decision_to_its_own_inbox_record():
    text = daily_digest.format_slack_digest(
        date="2026-11-15",
        agents=[],
        assignments=[],
        loops={},
        tasks=[],
        inbox_counts={},
        products=[],
        customers=[],
        inbox_items=[{"id": "capture/1", "source": "slack", "text": "Approve the pilot?"}],
    )
    assert "Approve the pilot?" in text and "#/inbox/capture%2F1|Open idea>" in text
