# Sprint: Agent Studio UX overhaul

Status: scoped, not started. Design canvas: private claude.ai artifact (link in
the PR or session notes; not embedded here because it is not public).

## Goal

Make the Studio answer one question fast: **what needs me right now, and what
is the Mini doing?** Then let the operator act on it in one place. This follows
the flow already stated in `STUDIO-SWITCHBOARD-ARCHITECTURE.md`: overview, select
work, inspect evidence, guide action.

Non-goals: new backend capabilities, a new component library, light mode,
a custom typeface, or changing the Map's graph interaction model.

## Evidence (verified on `main` at `92ee396`)

Method: built `dashboard/`, served `dist/` in Chromium with every `/api/**` call
and the `/ws` socket mocked (same technique as `tests/studio-ux.browser.mjs`),
screenshotted all 7 tabs at 1440px and 390px, and counted computed font sizes.
Baseline `npm run lint`, `npm test`, and `npm run build` pass (one pre-existing
lint warning in `ErrorBoundary.tsx`).

| # | Finding | Where |
|---|---|---|
| 1 | "Needs attention" has two definitions and shows two numbers. Today counts output awaiting review (6 in the fixture); Overview counts approvals plus failures (1). | `assignmentReview.ts`, `ActivityOverview.tsx:182` |
| 2 | Three overview-style pages (Today, Overview, Map), each with its own header, metrics, and polling. "Runs today" is UTC on Today and local time on Overview. | `Switchboard.tsx`, `ActivityOverview.tsx:332` |
| 3 | The tab bar is rendered separately by each view (6 call sites) and jumps position between tabs. The header style differs on every view. | `StudioTabs.tsx`, each view |
| 4 | Agent run is in 3 places, config editing in 2, loop toggle in 3. | `AgentStudio.tsx`, `AgentInspector.tsx`, `ActivityOverview.tsx`, `Switchboard.tsx` |
| 5 | Small text: 53% of text elements on Today and 65% on Overview render at 11px or less. 160 arbitrary `text-[8-12px]` classes in 20 files. | computed styles; grep |
| 6 | Contrast: `text-neutral-500` (84 uses) measures about 4.2:1 on `neutral-950` and fails AA at small sizes. `neutral-600` (45 uses, about 2.5:1) and `neutral-700` (18 uses) fail at any size. `design/tokens.json` lists neutral-500 as `text.subtle`, so the token itself needs a usage rule. | grep; WCAG contrast formula |
| 7 | Tokens not adopted: tokens say emerald accent and rose danger. The UI uses cyan as primary, purple for brand, fuchsia on the Map, red for danger. 11 color families plus 87 hex values in `switchboard.css`. | `design/tokens.json`, `switchboard.css` |
| 8 | Keyboard bug: with voice configured, Space on a focused button starts the mic and `preventDefault` blocks the click. | `VoiceConductor.tsx:71-78` |
| 9 | Blast radius: one error boundary wraps nav and content together. A single malformed record (reproduced with a handoff missing `instructions`/`artifact`) replaces the whole Studio, tabs included, with "Agent Studio crashed". | `Console.tsx:16`, `ActivityOverview.tsx:132,367` |
| 10 | Hardcoded state: "Mini online" is always green; model label `qwen3:8b` is hardcoded. | `AgentStudio.tsx:204,216` |
| 11 | No routing. Views switch via `useState`; cross-view navigation uses a window `CustomEvent` plus key-remount "intent + revision" state. No deep links, back button does nothing. | `AgentStudio.tsx:50-116`, `Pulse.tsx:60` |
| 12 | Four orphaned components, 1,260 lines, zero importers: `RightNow`, `Direct`, `Mind`, `Files`. `RightNow` is the only UI for action approve/reject, so that feature is unreachable. | `console/` |
| 13 | Voice Conductor and Pulse footers take about 130px of height on every view, including when voice is not configured. | `Console.tsx:20-26` |
| 14 | Two independent `/ws` sockets (Overview, Map) and 5s polling of the same `assignments` key from 4 components. | `ActivityOverview.tsx:73`, `AgentWorldCanvas.tsx:202` |
| 15 | CI runs lint, unit tests, and build for the dashboard, but none of the 7 `tests/*.browser.mjs` Playwright suites. | `.github/workflows/ci.yml:13-29` |

## Target information architecture

7 tabs become 5 destinations in one persistent shell (left rail on desktop,
bottom bar under 640px):

| New | Replaces | Job |
|---|---|---|
| **Today** | Today + Overview | One "Needs you" queue (failed, ready for review, approvals), running now, recent activity, loops. The activity ledger lives here. |
| **Inbox** | Memory | Triage Slack captures and loop proposals. It is an inbox, so name it one. |
| **Work** | Assignments + Handoffs | List and detail. A handoff is a relation on a work item and a filter, not a separate page. |
| **Agents** | Agents | Roster plus an agent detail page (Overview, Configure, Runs, Usage). The only place config is edited. |
| **Map** | Map | Unchanged interaction, inside the shell. Inspector becomes a peek that links to agent detail. |

System status (Brain, Ollama, Mini slot, loops, voice) moves into the rail
footer. Pulse is removed as a separate footer.

Routes: `#/today`, `#/inbox`, `#/work/:id?`, `#/agents/:id?/:tab?`, `#/map`.
Hash routing needs no server config and no new dependency.

## Design rules

- Tokens from `design/tokens.json` become Tailwind v4 `@theme` variables in
  `index.css`. Accent emerald, status emerald / amber / rose / indigo.
- Status is always icon plus word. Never a bare colored dot.
- Type floor 12px. Scale 12 / 13 / 14 / 16 / 20 / 24. Sentence-case labels
  instead of 0.28em tracked uppercase eyebrows.
- Secondary text `neutral-400`. `neutral-500` only at 18px+ or decorative.
  Never `neutral-600`/`700` for text.
- Every interactive element gets a 2px emerald `focus-visible` ring.
- Row targets 32px on desktop, 44px on phone.
- One primary action per row or panel; secondary actions behind a menu or
  disclosure.

## Delivery plan

Each PR is independently shippable and reviewable. Order is by risk reduction
first, then user-visible value.

### PR 1: `fix(studio)`: correctness and blast radius (small)

- Scope error boundaries per view, with nav outside them (finding 9).
- Guard `cleanSnippet` and siblings against missing strings.
- Fix the Space key handler to skip buttons, links, and `[role]` targets, and to
  only `preventDefault` when it actually starts or stops talking (finding 8).
- Replace hardcoded "Mini online" and `qwen3:8b` with live data (finding 10).
- One `needsAttention()` in `assignmentReview.ts` used by both Today and
  Overview, with a unit test (finding 1). One timezone rule: UTC, labelled.
- Tests: unit test for `needsAttention`; browser test that a malformed handoff
  leaves the tabs usable; browser test for Space on a focused button.

### PR 2: `ci`: run the browser suites

- Add a CI job that builds `dist/` and runs `tests/*.browser.mjs` with
  `STUDIO_DIST_DIR` (all APIs mocked, no backend). Upload screenshots as an
  artifact on failure.
- Add an axe-core pass on each view as a non-blocking report first; make it
  blocking after PR 4.
- Rationale: every later PR changes layout. Without this, regressions ship.

### PR 3: `refactor(studio)`: shell and routing (no visual redesign yet)

- `StudioShell` renders nav once; views render into it. Delete the 6
  `StudioTabs` call sites and the per-view page headers in favor of one
  `PageHeader`.
- Hash router (small, in-repo) replaces `useState` view switching, the
  `studio:switchboard` CustomEvent, and the key-remount intents.
- One `useStudioSocket` hook shared by Overview and Map (finding 14).
- Tests: browser test for deep links and back/forward on every route.

### PR 4: `design(studio)`: adopt tokens

- `@theme` tokens; mechanical sweep of arbitrary font sizes, low-contrast
  greys, and off-token colors. Port `switchboard.css` hex values to tokens.
- `focus-visible` ring everywhere.
- Update `design/TODO.md` (it still says the dashboard has no Tailwind).
- Tests: extend the computed-style check from this scoping pass into CI:
  zero text under 12px, zero `neutral-600/700` text.

### PR 5: `feat(studio)`: Today

- Merge Overview into Today per the design canvas. Delete `ActivityOverview`
  as a route; its ledger becomes a Today section.
- Rail status block; remove Pulse footer; voice becomes a rail control.

### PR 6: `feat(studio)`: Agent detail

- Roster plus `#/agents/:id/:tab`. Map `AgentInspector` becomes a read-only
  peek with "Open agent". Remove duplicate run/config paths.

### PR 7: `feat(studio)`: Work

- Merge Assignments and Handoffs. Split `WorkBoard.tsx` (735 lines) into list,
  detail, and panels. Secondary panels (history, git workspace, context)
  collapse behind disclosures; review actions are primary.

### PR 8: `feat(studio)`: Inbox

- Rename Memory to Inbox. One expanded row at a time, primary "Queue as work",
  "Log to memory", and a "Close as" menu that holds the four triage outcomes.

## Decisions needed

1. **Orphaned components.** Recommend deleting `Direct`, `Mind`, `Files` in
   PR 1. `RightNow` holds the only action-approval UI: either confirm that
   approvals are handled elsewhere and delete it, or fold approvals into the
   Today "Needs you" queue in PR 5.
2. **Accent color.** Recommend emerald, because `design/` names aiia-console
   canonical. The alternative is to update `tokens.json` to cyan and keep the
   current look. Either is fine; two accents is not.
3. **Voice Conductor.** Recommend moving it to a rail control that is hidden
   when voice is not configured.

## Risks

- Browser tests hard-code current copy and tab names. PRs 3 and 5-8 will
  rewrite them; do that in the same PR as the UI change, never after.
- The review inbox and assignment review flows carry real audit semantics
  (see `STUDIO-OUTPUT-RECOVERY.md`, `STUDIO-ASSIGNMENT-HISTORY.md`). Layout
  changes must not change which API call a button makes.
- The sprint is frontend only. Any PR that needs a backend change stops and is
  split.

## Backlog (not this sprint)

- Light mode and resolved hex tokens (`design/TODO.md`).
- Custom typeface.
- Per-agent durable token attribution (backend slice, see
  `STUDIO-USAGE-MAP-2026-09-15.md`).
- Stale references in `SPRINT-agent-studio-world-canvas.md`
  (`VoidStarWorld.tsx`, `voidstarProjection.ts` no longer exist).
