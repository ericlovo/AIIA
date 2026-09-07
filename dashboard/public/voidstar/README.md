# Vendored Void Star renderer

`city.html` is copied from `voidstar-site/city.html` and loaded in a same-origin
iframe by Agent Studio. The PWA metadata and service-worker registration are
removed because the dashboard owns the outer application shell.

Keep the `window.VOIDSTAR` and `window.__vsGroundPoints` integration surface
compatible when refreshing this file from the canonical Void Star project.

The React overlay owns interaction state. Agent and assignment geometry is
persisted through `/api/agent-world/layout`; handoff ports route into the
existing guarded handoff form. Only privacy-bounded lifecycle summaries cross
the renderer boundary—never prompts, tokens, code, or work-product contents.
