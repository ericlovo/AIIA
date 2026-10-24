# Studio connection recovery

Studio's main research view is `/#/bd`. Review delivered packets there; inspect
source evidence before choosing Research next, Watch or Dismiss. Those choices
only update local review metadata. They do not send outreach. The queue fills
when a scheduled cycle is due; an empty queue between cycles is normal.

`/#/projects` shows engineering checkout and CI evidence, not the BD pipeline.

## Public URL versus local origin

The Mini's tunnel configuration maps `aiia.getsanction.com` to the loopback
Command Center on port 8200. Frontend API requests use the current origin.
A successful local request does not prove the public browser's authenticated
session works. Cloudflare Access can require a fresh top-level sign-in even when
an already-open Studio shell still renders. Network failure alone does not prove
an expired session: connectivity and service failures can look similar.

Use the recovery actions to retry requests or reload the current Studio page.
The reconnect action stays on the current origin and preserves the current view;
it never accepts an arbitrary external return URL. If the public sign-in flow is
required, complete that flow in the browser. Do not disable Access, copy browser
cookies into scripts, or expose the loopback service to work around sign-in.

The HTML entrypoint is served with `Cache-Control: no-store` so future page loads
receive the current asset manifest. This does not remotely replace JavaScript in
an already-open tab, nor does it prove a previously observed old shell was cached.

When verifying a fix, test the public URL through its authorized sign-in flow,
not only localhost. Check the requested page, API-backed content, primary
navigation and the packet-review action. Report public verification as blocked
if authentication cannot be completed; do not call a local-only result a public
fix.
