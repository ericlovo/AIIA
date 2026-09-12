# Assignment Attempt History

Saturday slice, 2026-09-12. Built from merged PR #55 (`980211d`) on
`feat/studio-assignment-history`.

Assignment and downstream handoff details now provide a paged timeline of saved
attempts. Opening an attempt fetches its original task, output and error from the
existing run ledger. Model, duration, recorded time and imported provenance are
shown, and the run applied to the assignment is identified. Current review status
remains attached to the assignment; a successful run is not labeled accepted.

`GET /api/assignments/{id}/history?offset=0` returns 20 saved attempts per page,
total count, latest attempt identity and whether its output is recorded. Queries
are scoped to assignment-triggered runs for that assignment, have stable ordering
by timestamp and ID, and do not inherit the switchboard's 91-day window. A SQLite
index supports the lookup. Full output is fetched only for the expanded attempt.

Missing current output is shown explicitly, including when prior saved attempts
exist. Legacy records without attempt identity say so. Storage failure returns
503 and the UI shows unavailable history, rather than treating it as zero runs.
History refresh uses the existing durable outbox replay; it does not invoke models
or change assignment/review state.

This is saved-attempt history, not a complete worker event log. Attempts with no
durably saved run cannot be reconstructed. Offset pages are a live view; newly
completed attempts may shift records between pages while operators browse.

Validation: 404 backend tests passed, 9 skipped, one existing macOS MIME failure.
Focused tests cover scoping, old records, pagination, missing output and storage
errors. Frontend production build and lint pass (one existing directive warning).
Synthetic browser checks at 1440x1000 and 390x844 cover expanded evidence,
pagination and explicit errors, with zero mutation requests. Screenshots inspected.

Deployment is pending. No live agents, schedules, or model calls were used for
this slice.
