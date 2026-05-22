# Design Workstream — Deferred

These are out of scope for the design+desktop launch pass and tracked here so they don't get lost.

## Dashboard restyle (this repo, `dashboard/`)

The Vite dashboard in this repo hasn't adopted Tailwind or the tokens documented in `tokens.json`. When that workstream picks up:

- Add `tailwindcss@^4` + `@tailwindcss/vite` to `dashboard/package.json`
- Add `@import "tailwindcss";` to `dashboard/src/main.css` (or equivalent)
- Mirror the surface/text/border/accent classes used in aiia-console (see `tokens.json`)
- Adopt the wordmark `⌬` consistently in dashboard headers

## Token formalization

`tokens.json` v0.1 uses Tailwind named colors (`neutral-900`) rather than raw hex. When the system needs to leave Tailwind's color space (e.g. for a future signed macOS app's NSColor bridge, a brand-color shift, or a light-mode pass):

- Resolve named tokens to hex (lock the version of Tailwind)
- Add a `mode: light` palette alongside the current `mode: dark-first`
- Add custom radii + spacing scales if/when components diverge from Tailwind defaults

## Typography

System font stack is fine for v0.1. If we ship a custom face (e.g. for the wordmark or display headings), document it here and in `tokens.json` under `typography`.
