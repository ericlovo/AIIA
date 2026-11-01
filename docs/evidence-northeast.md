# Northeast public-source checks — 2026-10-31

These are source-level observations, not a claim of statewide school-district coverage. No credentials, provider calls, Slack delivery or TLS bypass were used.

## Vermont — working, agency-scoped collector

- Public state bid list: https://www.vermontbusinessregistry.com/bidsearch.aspx?type=5
- The Vermont legislature's official system inventory identifies VermontBusinessRegistry as the state/municipal bid board hosted by ACCD/ADS: https://legislature.vermont.gov/Documents/2020/WorkGroups/JITOC/Documents%20and%20Testimony/W~John%20Quinn~System%20Interaction%20Inventory~11-15-2019.pdf
- HTTP 200 and actual solicitation records verified with certificate validation. Public ASP.NET pagination requires its normal anonymous session cookie and form state; a stateless pagination POST redirected, while the same public client session returned the second page. No login is involved.
- `scripts/education_northeast.py` traverses all advertised state-list pages, selects Agency of Education notices and verifies each detail title, agency and closing date against its listing. Seven-request total limit; changed/repeated pagination or exhausted budget fails rather than claiming complete coverage.
- Live result: three qualifying RFPs, after checking full visible detail context for cancellation:
  - Chronic Absenteeism Implementation Support, source record 77251, due November 18: https://www.vermontbusinessregistry.com/BidPreview.aspx?BidID=77251
  - Academic Excellence for All PreK-12 Implementation, source record 76199, due November 20: https://www.vermontbusinessregistry.com/BidPreview.aspx?BidID=76199
  - Vermont Seal of Global Leadership, source record 76194, due November 4: https://www.vermontbusinessregistry.com/BidPreview.aspx?BidID=76194
- Dates above are 2026; time zones are not inferred. Source record IDs are not fabricated solicitation numbers. Contact fields are excluded. Attachments are not downloaded or checked; alerts remain review notices linked to the official record.

## Maine — VSS client adapter required; legacy page is insufficient

- Official procurement page: https://www.maine.gov/dafs/bbm/procurementservices/vendors/rfps
- Primary search-index evidence explicitly says new solicitations moved to Vendor Self Service on October 1, 2025. The static RFP table is legacy, not current coverage.
- Direct procurement-page probes returned HTTP 403 CloudFront with both the default client and an honest AIIA audit user agent. No bypass attempted.
- Official VSS routing evidence: https://www.maine.gov/dafs/bbm/procurementservices/vendors/vendor-self-service-system
- Published portal https://mevss.hostams.com/PRDVSS1X1/Advantage4 returns HTTP 200 with verified TLS, but only a JavaScript application shell (12.6KB), no solicitation records. Requires bounded public API discovery and verification of anonymous search, agency/type/status filters and pagination. Do not label the shell as no matches, and do not assume an account is required merely because submission requires one.
- No Maine collector was added.

## New Hampshire — education host access blocked

- https://www.education.nh.gov/ returned HTTP 403 Akamai Access Denied using both default client and honest audit user agent.
- Bounded official-source search did not identify a verified current education listing available to this collector. Old indexed RFP PDFs and proposed legislation are not evidence of current opportunities.
- Dependency: a permitted public department feed/listing or reliable official central-procurement route with verified NHED records. No proxy, user-agent impersonation, alternate TLS trust or login bypass attempted.
- No New Hampshire collector was added.

## Rhode Island — public portal discovered, TLS dependency unresolved

- Official route https://ridop.ri.gov/vendors/bidding-opportunities returns HTTP 200 with the honest audit user agent and embeds https://webprocure.proactiscloud.com/wp-web-public/#/bidboard/search?&customerid=46&oid=-1
- Official FAQ says the bid board permits public viewing without registration: https://ridop.ri.gov/ocean-state-procures-osp/faqs
- Direct Proactis probe fails certificate verification (`unable to get local issuer certificate`) with default verified TLS; no verification disabling or custom trust was used. Need a valid chain in the deployment trust environment, then public API and pagination verification for customer 46.
- The separate official external-board search https://purchasing.ri.gov/bidding/externalbidsearch.aspx returns HTTP 200 with ASP.NET search controls, including school districts and active/awarded/canceled status choices. It is a search form, not a verified result set. A bounded form/result adapter could cover explicitly identified external buyers, but would not substitute for RIDE state-agency coverage.
- No Rhode Island collector was added.
