# Studio Usage and Map UX

## Scope

- Switchboard now displays reported platform input/output tokens and request
  totals, provider totals, purpose attribution, and fourteen UTC days of history.
- Uses existing `/api/tokens/today` and `/api/tokens/recent` endpoints; refreshes
  every fifteen seconds and supports manual retry. Failed refreshes retain and
  label previously loaded totals.
- No new inference requests, token estimates, or model pricing were introduced.
- Map layout grows beyond the previous fixed grid capacity, reserves enough
  width for nodes, and reflows collisions instead of stacking nodes.
- Map header wraps in narrow panels. Completed-work visibility, reset, zoom,
  and fit controls stay outside the scrollable graph. The node inspector and
  handoff target prompt remain in the visible viewport.
- Map loading and partial-data failures are explicit. Active count represents
  running agents, without counting their running assignment a second time.

## Boundaries

Token reporting is currently best-effort. These totals are not a billing ledger.
Studio purposes combine multiple agents; per-agent and per-run durable usage
attribution remain a separate backend slice. Historical missing reports cannot
be reconstructed from token caps. Local inference has no per-token API charge,
but still consumes Mini capacity.

No production restart, deployment, GitHub push, live layout reset, agent run,
assignment mutation, or Slack configuration change was performed for this slice.

## Verification

From `dashboard`:

```sh
npm test
npm run build
npm run lint
node tests/studio-ux.browser.mjs
```

The browser script requires Playwright and its Chromium browser. It defaults to
`http://127.0.0.1:5184/`; override `STUDIO_URL` as needed. `PLAYWRIGHT_MODULE` can
point to an existing Playwright module; `CHROMIUM_PATH` optionally selects a
browser executable. `SCREENSHOT_DIR` overrides the temporary screenshot folder.

Every API response and the Studio WebSocket in this test is intercepted with
synthetic data. Tests never send writes to the Mini. Coverage includes:

- 1440px, 653px, and 390px viewports;
- token display, purpose breakdown, refresh failure, recovery, and zero usage;
- 48- and 96-node maps without overlapping node rectangles;
- fit, viewport-contained inspector, keyboard position saving, save failure,
  and reset recovery;
- no uncaught browser errors or document-level horizontal overflow.

Unit tests cover collision-heavy layouts through 160 nodes, deterministic
placement, invalid positions, and preservation of valid positions.

Results: 14 unit tests passed; browser checks passed at all three widths;
production build passed; lint reported only the pre-existing unused
eslint-disable warning in `ErrorBoundary.tsx`. Backend code was unchanged and
the backend suite was not rerun for this frontend slice.
