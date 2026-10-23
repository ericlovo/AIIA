# Combined synthetic shadow run

This explicit workstation CLI joins Mini-local Qwen extraction, workstation Jev
judgment and Managed Leads shadow intake. It is not a scheduled job or a mounted
Studio feature. Only the fixed invented Cedar Works sentence is used. No source
fetching, production leads, approvals, accounts or deliveries are created.

The Mini keeps its existing airgap. Qwen uses loopback Ollama through existing
SSH; Jev runs from the workstation using its separately held credential.
`typesafe.signals` must remain disabled on the Mini. The complete airgap map is
compared before and after inference and on completion. No service restart,
configuration edit, model installation or scheduler change is performed.

## Invocation

Run from the reviewed checkout, with a new private directory under an existing
private parent. Supply the existing synthetic workspace and runner credential;
the explicit scope flag is an operator attestation, not a server-side proof that
a workspace contains no production data. Independently audit the workspace.

```sh
python -m local_brain.scripts.combined_synthetic_shadow \
  --run-dir /private/path/new-run \
  --expected-workspace SYNTHETIC_WORKSPACE_UUID \
  --synthetic-scope-confirmed \
  --mini-host EXISTING_SSH_HOST \
  --mini-python /absolute/runtime/bin/python \
  --mini-repo /absolute/AIIA-checkout \
  --runner-credential-file /private/path/runner.json \
  --jev-credential-file /private/path/jev.json
```

Credential and artifact files are operator-owned mode 0600 in mode 0700 directories.
Never commit their contents. The run manifest binds the workspace, source, Mini
endpoint, models and policy. Qwen's extracted claims and exact source span feed
Jev's request; fabricated or malformed spans are rejected. Jev is pinned to
`jev-1.13.0`. Policy `shadow-v1` retains probability/link thresholds of 0.8.

A valid abstention or withholding judgment can enter the **shadow ledger** with
an explicit correction-needed gap. It never means a qualified or approved lead.
All submitted candidates have unknown identity, pending review, no verified
claims and no contacts. Malformed provider answers block submission.

## Recovery and evidence

Private artifacts retain the extraction, bound Jev request, provider response,
judgment, assignment, original submission and receipt. Exclusive pending markers
are written before provider calls. A marker without a completed result blocks an
automatic retry: investigate the uncertain call rather than deleting markers.
Successful saved provider results resume without another inference.

An uncertain staging request may be retried with the identical saved envelope
and idempotency key. Never manufacture a replacement key to hide uncertainty.
The CLI reads the immutable receipt back and compares it to the response. It
checks the complete airgap map again even when a later step fails.

Verify accounts, people, signals, operational research runs and deliveries using
a separate authorized operator audit before and after. Runner credentials cannot
perform that audit. A successful shadow receipt proves transport and isolation,
not model quality or commercial lead fitness.

## Observed synthetic result — October 22, 2026

One Mini Qwen call extracted the exact invented company, Wisconsin location and
`expanding` event in 6,636.7 ms (59 input / 45 output tokens). The recorded Jev
response took 376.2 ms (937 input / 138 output tokens). Probabilities were 0.94
support, 0.98 organization, 0.95 location, 0.97 event and 0.83 relevance, with
linked evidence confidence 1.0. The unchanged policy returned
`supported_candidate`; the submitted candidate remained pending and unverified.
These are single-fixture measurements, not a quality or performance benchmark.

Managed Leads returned 201 / `shadow_received` with one candidate. An exact
submission retry returned 200 / replayed with the identical receipt, and receipt
lookup matched. Independent authenticated operator audits before and after found
zero accounts, people, signals, operational research runs and deliveries. The
complete Mini airgap snapshots matched. No schedule or production lead was added.

The exercise exposed two client integration issues: the existing Jev credential
parent was not private enough, and an edge response cookie was incorrectly
rejected by the ephemeral provider client. The credential was copied privately
without changing the existing directory. The cookie check was corrected and
regression-tested; the closed single-request client never reuses cookies. One
explicit Jev recovery request followed the discarded response, whose inference
outcome is unknown. Qwen ran only once. Pending markers and recovery explanations
remain with the private artifacts; no provider response is represented as a
replay fixture or a production lead.
