# New York source verification

Verified 2026-10-31 with HTTPX, normal TLS verification, `trust_env=False`, and no redirects or proxy workaround.

- NYSED competitive service contract procurement page failed certificate verification: unable to get local issuer certificate. This adapter does **not** repair or bypass that chain failure.
- Official alternative [NYC DOE request for proposals](https://infohub.nyced.org/in-our-schools/working-with-the-doe/contracts-and-purchasing/open-doe-solicitations/request-for-proposals) returned HTTP 200 with 210,762 bytes of HTML. Its individual R-number headings contain descriptions and explicitly labelled proposal deadlines.
- Parsing the retrieved page produced one future dated notice: R1894, Public School Athletic League insurance and benefits, proposal deadline December 9, 2026. Historical deadlines, unknown deadlines, and indefinite solicitations were withheld. This is parser evidence, not a Jev judgment or eligibility endorsement.
- [NYS Contract Reporter](https://www.nyscr.ny.gov/Ads/Search) also returned HTTP 200, but no statewide buyer/notice adapter was implemented here.

## Integration

Use adapter `newyork` with the NYC DOE source URL above. The callable is `fetch(client, source, today)`; today is an ISO date. Returned state is NY and each record includes `notice_number`, `notice_id`, `deadline_date`, and `coverage_scope` in addition to normal candidate fields.

All notices share the actual official listing URL. Preserve the notice number in delivery identity when normalizing the state: hash state + URL + notice_number. Hashing only state + URL would incorrectly collapse distinct solicitations. No fabricated per-notice fragment or URL is used.

This provides **New York City district coverage only**, not statewide NYSED coverage or all New York districts. Keep NYSED's original failure visible if retained as a separate source. Count a successful retrieval as a district source, not a repaired statewide feed.

The parser binds evidence to a single heading section. It rejects challenges, missing notice structure, duplicate notice numbers, oversized responses, redirects and unexpected content types. Explicit cancellation/closure terms exclude records. Only a unique labelled future proposal deadline qualifies; same-day deadlines are withheld because time zone parsing is not implemented. Conflicting deadline updates and sections over 12,000 characters are withheld. Jev still performs subsequent relevance screening.
