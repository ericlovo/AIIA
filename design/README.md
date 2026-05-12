# ⌬ AIIA Design

This folder holds the source-of-truth design tokens for the AIIA product surface.

## Contract

- **Canonical implementation**: [`ericlovo/aiia-console`](https://github.com/ericlovo/aiia-console) (Tauri 2 + React 19 + Tailwind 4.3). Whatever ships in the desktop console is the reference; this folder documents it.
- **Mirror**: The Vite dashboard in `dashboard/` (this repo) should match these tokens when restyled. As of v0.5.0 it hasn't been restyled yet — see `TODO.md`.
- **Update flow**: when the console's theme changes, update `tokens.json` here in the same PR (or a follow-up). Don't let this drift silently.

## What's in `tokens.json`

A v0.1 snapshot of the de facto design language extracted from how aiia-console actually styles its components:

- Dark-mode-first neutral surface palette (neutral-900 base, 950 deepest, 800 raised)
- Emerald as the primary accent (`ring-emerald-500` is the focus ring, used 22 times in v0.1)
- Amber / Rose / Indigo for warning / danger / info
- System font stack, Tailwind default spacing + radii (no custom scales yet)

Values reference Tailwind's named palette (`neutral-900`, `emerald-500`) rather than raw hex so updates flow with the framework.

## Wordmark

`⌬ AIIA` is the wordmark, used in CLI startup, README headers, and the console title bar. Documented separately in the repo root (the README references it directly).
