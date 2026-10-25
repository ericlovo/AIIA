# Studio task and deployment verification

Daily Brief now requests at most 1024 output tokens, disables reasoning and session
retrieval, and bounds the supplied commit/health/token sections. The Brain forwards
the requested token limit to Ollama. Its existing 300-second deadline and the
caller's 330-second deadline are unchanged. A timeout returns HTTP 504 with a
specific error; there is no automatic retry or fabricated successful briefing.

The scheduled test runner requires the `asyncio` pytest plugin and strict pytest
configuration. Missing plugins, interrupted runs and non-test exit codes fail the
task. Actual assertion failures also fail the task, retain the final report and
record test trends. Failed results are not written as successful memories. A
later success replaces the failure normally; historical results are not rewritten.

Use a test environment installed with this repository's `.[dev]` dependencies.
`AIIA_TEST_PYTHON` may select its absolute Python executable separately from the
Brain runtime. Without that setting, the runner uses its current interpreter.
This change does not install packages, change schedules, or rerun failed jobs.

## Verified deployment receipts

GitHub merges and green CI do not prove deployment. Studio's morning note and
Slack digest can read an operator-owned JSON file selected by the
`AIIA_DEPLOYMENT_RECEIPTS` environment variable. It contains one **current** receipt
per exact GitHub repository. No file means unverified, not a deployment failure.

```json
{
  "version": 1,
  "repositories": {
    "owner/repository": {
      "state": "verified",
      "environment": "local-production",
      "sha": "0123456789012345678901234567890123456789",
      "verified_at": "2026-10-25T20:00:00Z",
      "health_passed": true,
      "browser_passed": true,
      "evidence": "/absolute/path/to/verification-report.json"
    }
  }
}
```

Write a receipt only after independently confirming the deployed revision,
service health, and rendered acceptance flow. Keep the supporting report at the
referenced location. The example is a schema example, not deployment evidence.
Receipts are trusted operator attestations, not independent live probes. Keep the
file outside source control, writable only by the deployment operator. Replace it
atomically. Before another deployment or rollback, replace the repository entry
with `state: pending`; use `state: failed` if verification fails. Do not retain an
older successful entry as current after a failed replacement.

Only `production` and `local-production` qualify. Receipts expire after seven days
and must have an aware timestamp that is not in the future. Missing, malformed,
expired, preview, pending or failed receipts never qualify. The mounted local Git
history must prove each PR's **merge commit** is an ancestor of the verified
revision. Missing objects or remote-only repositories remain unverified; this
collector does not fetch commits or probe services. Squash merges work through the
merge commit SHA, not the PR's branch SHA.

Studio shows verified and unverified merges separately even when reviews are
pending. Verification is evidence at the receipt timestamp, not a promise of
current uptime. Existing deployments need receipts based on retained verification
reports; this PR does not invent receipts for them or modify the live runtime.

Healthcare cadence, job limits, budgets, kill state and legacy signal flags are
outside this change. No additional research, model requests or outreach are run.
