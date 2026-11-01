# Optional USAC school technology discovery

Verified 2026-10-31. Not activated in worker, schedule, Jev, or Slack.

Official provenance: [USAC tools](https://www.usac.org/e-rate/resources/tools/) links the FCC Form 470 Tool Data. [Dataset metadata](https://opendata.usac.org/api/views/jt8s-3q52.json) and [resource API](https://opendata.usac.org/resource/jt8s-3q52.json) returned HTTP 200 with validated JSON structures, normal TLS verification, and no proxy workaround.

`collect(client, state, today)` selects only school and school district applicants with Certified status and an RFP document reference, certified within the preceding 90 days. It selects no contact names, personal addresses, telephone numbers, or email fields. Consortium and library records are excluded. Bounded pagination allows eight pages of 250 rows; a full final page raises an incomplete-source error instead of returning partial success. A successful query is category discovery, not general state procurement coverage.

All returned records are explicitly pending document review, not alert eligible, with unknown open status and unknown deadline. `allowable_contract_date` is retained as earliest permissible contracting date, never as bid deadline. No 28-day-derived open/closed claim is made. Attachment URLs are checked for the official publicdata.usac.org host and matching Form470 application path; attachment bytes/content have **not** been verified. An RFP-document field may point to a spreadsheet rather than a complete RFP. Official source records alone do not prove an open solicitation.

Live per-state grouped API counts below use the same Certified/school/RFP-reference criteria and certification interval 2026-08-02 inclusive to 2026-11-01 exclusive. These are raw service/document rows, not unique opportunities, approved leads, or verified open RFPs. States absent from the grouped result are represented as zero matching rows. This grouped query validates all-state queryability, not full per-state attachment retrieval. A separate real WI collect call returned 106 pending records. The adapter has not downloaded every state's records.

```json
{"AK": 6, "AL": 7, "AR": 451, "AZ": 210, "CA": 1076, "CO": 125, "CT": 44, "DE": 0, "FL": 83, "GA": 98, "HI": 4, "IA": 8, "ID": 25, "IL": 152, "IN": 343, "KS": 188, "KY": 27, "LA": 45, "MA": 8, "MD": 9, "ME": 27, "MI": 664, "MN": 50, "MO": 318, "MS": 213, "MT": 0, "NC": 52, "ND": 6, "NE": 16, "NH": 2, "NJ": 107, "NM": 51, "NV": 51, "NY": 179, "OH": 285, "OK": 1216, "OR": 165, "PA": 41, "RI": 24, "SC": 85, "SD": 18, "TN": 2, "TX": 1758, "UT": 99, "VA": 118, "VT": 2, "WA": 136, "WI": 106, "WV": 0, "WY": 2}
```
