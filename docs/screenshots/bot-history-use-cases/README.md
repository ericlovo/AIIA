# Bot history and use-case verification

Synthetic data, intercepted APIs, and blocked external requests. No live bot or
model execution. Captured from the production build at 1440×1000 and 390×1000.

| View | Desktop | Mobile |
|---|---|---|
| Last activity on cards | [Desktop](bots-home-1440.png) | [Mobile](bots-home-390.png) |
| Use cases and additional context | [Desktop](bots-detail-1440.png) | [Mobile](bots-detail-390.png) |
| Expanded completed and failed history | [Desktop](bots-history-1440.png) | [Mobile](bots-history-390.png) |

Run `npm run build` then `node tests/bots.browser.mjs` from `dashboard/` to
reproduce. Assertions cover immediate shortcuts with/without context, free-form
submission, the combined length limit, in-flight guards across bot navigation,
draft preservation, history ordering, expansion, pagination, empty/error/retry
states, and viewport overflow. Backend tests separately exercise the real API
against temporary SQLite/config files, including more than 12 runs and restart.
