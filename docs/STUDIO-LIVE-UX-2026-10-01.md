# Studio live UX audit - 2026-10-01

## Deployed and observed

- PR #94 deployed on the Mini at `3c2d3b1`, including timelines, lead qualification, research snapshots and the lead queue.
- Read-only browser audit visited Today, Overview, Agents, Assignments, Handoffs, Memory, Signals and Map at 1440px and 390px widths. No page exceptions or horizontal document overflow. This is navigation evidence, not proof of all mutation flows.
- Live lead endpoint returned 200 with zero leads. Discovery and screening were disabled; the credential was configured. Both discovery schedules were disabled.
- Today showed 80 items needing review. The DOM contained 382 buttons, including activity controls; this count is not a count of simultaneously visible controls.
- Brain and Ollama were online; the optional default/platform services were offline. TopBar requested `/api/autonomy/status`, which returned 404 and silently fell back to `phase1`.
- Mobile persistent voice and Pulse controls occupy substantial vertical space. The live voice integration is configured, so simply hiding an unconfigured integration would not address it.

## This follow-up

- Lead queue before discovery configuration.
- Discovery controls collapsed by default, with scheduled job count in the disclosure.
- Distinguish a configured but disabled Jev integration from a missing credential.
- Explain disabled retrieval/screening within discovery configuration; keep run controls disabled until the server reports readiness.
- No changes to egress flags, schedules, review decisions, approval gates or outreach.

## Next slices

1. Verify watch-loop outcome classification and stop routine no-change runs from creating new review obligations. Preserve historical review items until explicitly triaged.
2. Replace unsupported autonomy phase fallback with verified status or remove it.
3. Compact idle voice and system status into accessible disclosures while preserving recording indicators and keyboard safeguards.
4. Enable bounded public discovery only with explicit operational opt-in, then validate source evidence -> qualification -> queued research on real data. Existing bounds are three items per run, six pending and twelve-hour intervals.

## Deployment recovery

The previous Homebrew Python runtime failed to import `pyexpat` because of an Expat symbol mismatch. The initial restart incorrectly proceeded despite a failed import gate. Recovery used a separate managed Python 3.12.13 environment with the existing 147 installed package versions, then repointed the external launch script. The old environment was retained. Brain and Command Center recovered; this does not establish that unrelated scripts using the old environment are repaired.

Runtime databases, environment files, prior dashboard assets and launcher were backed up locally before changes. The dependency snapshot was retained with that backup. Future release commands must stop immediately on any failed preflight before copying assets or restarting services.
