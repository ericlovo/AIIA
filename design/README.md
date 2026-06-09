# ⌬ AIIA Design

This folder holds the source-of-truth design tokens for the AIIA product surface.

## Contract

- **Canonical implementation**: [`ericlovo/aiia-console`](https://github.com/ericlovo/aiia-console) (Tauri 2 + React 19 + Tailwind 4.3). Whatever ships in the desktop console is the reference; this folder documents it.
- **Mirror**: The Vite dashboard in `dashboard/` (this repo) should match these tokens when restyled. As of v0.5.0 it hasn't been restyled yet — see `TODO.md`.
- **Update flow**: when the console's theme changes, update `tokens.json` here in the same PR (or a follow-up). Don't let this drift silently.

## What's in `tokens.json`

A v0.2 snapshot of the de facto design language extracted from the console's
`src/App.css` `@theme` block (the layer that actually compiles to Tailwind
utilities):

- **Light-first vellum/ink brand surface** (Penguin Classics-inspired, Claude.ai-quiet) — warm cream `vellum-50` (#FBF7EC) page, graduated `carbon-*` cream→ink surfaces, warm near-black `ink-900` (#14110D) text
- **Serif type** — EB Garamond for body, Cormorant Garamond for display, JetBrains Mono for code
- **Cinnabar** (#C13B2A) as the reserved single-spot accent
- Saturated `status-*` colours (healthy / attention / failing / active / …) kept readable on cream
- Custom radius scale (4/8/12/16px) and letter-spacing scale (tight → brand 0.40em)

> Note: the console keeps the legacy utility names (`void`, `carbon-*`,
> `text-*`, `amethyst-*`) but **rebinds** their values onto the vellum/ink
> scale — so those names are historical and no longer describe colour. The
> tokens here capture the effective (rebound) values as raw hex.

## Wordmark

`⌬ AIIA` is the wordmark, used in CLI startup, README headers, and the console title bar. Documented separately in the repo root (the README references it directly).
