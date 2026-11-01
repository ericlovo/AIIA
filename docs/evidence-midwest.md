# Midwest public-source evidence

Observed 2026-10-31 America/Chicago (2026-11-01 UTC). All probes used verified HTTPS, bounded 10–20 second requests, no accounts, no messages, and no TLS bypass. A returned HTML page alone was not treated as collected notice coverage.

## Nebraska: implemented, pending qualification

- Source: https://das.nebraska.gov/materiel/bid-opportunities.html
- HTTP 200; the explicitly identified `section#current-bid-opportunities` contains one complete table: seven current records. Separate in-process and awarded tables are excluded.
- One future NDE RFP row: NDERFP260915, Supported Employment Services, listed opening November 13, 2026.
- Detail: https://das.nebraska.gov/materiel/purchasing/NDERFP260915/NDERFP260915.html
- Detail identifies NDE Office of Vocational Rehab, repeats **OPENING DATE**, and links a Word solicitation. No explicit HTML submission deadline was verified, and K12 applicability is not established.
- `education_midwest.fetch` returns one pending item with `alert_eligible=False`, `qualification_status=pending_submission_deadline_and_k12_verification`. It does not relabel opening date as submission deadline or claim this is a qualified K12 opportunity.
- Scope: current NDE agency rows on DAS; not exhaustive district coverage. Max 200 current rows, 2 MB response, 20-second fetch; schema changes, duplicate IDs, wrong hosts, redirects and missing current-table structure fail.
- Alternate education-agency listing https://www.education.ne.gov/legal/nde-contract-opportunities/ returns HTTP 403. DAS is the verified working collection path.
- Validation: live source collection returned one pending item; ten focused tests pass; Ruff clean.

## Wisconsin: records available, filtered collection blocked

- Public search: https://esupplier.wi.gov/psc/esupplier_2/SUPPLIER/ERP/c/WI_SS_SELF_SERVICE.WI_SS_BIDDER_BIDS.GBL
- HTTP 200, real notice rows with event number, explicit solicitation type, buyer, status and due date/time. First page showed 25 of 3,123 records, including archived statuses. First page is not complete coverage.
- A session-preserving, read-only PeopleSoft search POST for agency `Public Instruction` returned HTTP 200 XML with a server-side error: `GetField: Field 'WI_SS_BIDR_SRCH.WI_SS_MY_NIGP' not found in scroll level 0. (2,155) ... SEARCH.FieldChange ... Statement:51`. Returned partial rows were not accepted as a successful agency search.
- Dependency: working supported agency-filter/pagination response or another complete public listing. No credentials were requested or used.
- Official public-notice alternative: https://publicnotices.wi.gov/ redirects to https://publicnotices.wisconsin.gov/, which returned an unhydrated HTML shell. `www.publicnotices.wi.gov` has a certificate hostname mismatch and was not used. VendorNet `/Bids.aspx` returned 404.

## Kansas: actual rows, completeness not verified

- Public source: https://supplier.sok.ks.gov/psc/sokfsprdsup/SUPPLIER/ERP/c/SCP_PUBLIC_MENU_FL.SCP_PUB_BID_CMP_FL.GBL2?PAGE=SCP_PUB_BIDLIST_FL
- HTTP 200, PeopleSoft public bidding-event page with 50 actual events, agency names, event IDs, and opening/end date-times. Example: EVT0010980, Landscape and Maintenance - D1, Department of Transportation, December 10, 2026 (not K12).
- The page exposes `50 rows` but no independently verified overall total or complete pagination contract. Fifty rows may be a search cap. No healthy complete collector was inferred from this.
- Dependency: verify the public search's total/pagination or a complete education-agency filter and obtain explicit RFP/RFI type evidence. No adapter enabled.

## Minnesota: CAPTCHA, no procurement substitution

- Both the official old professional/technical listing at https://mn.gov/admin/osp/vendors/solicitations-and-contract-opportunities/professional-and-technical-solicitation-postings/ and new OSP domain https://osp.admin.mn.gov/ returned HTTP 200 **Radware Bot Manager Captcha**, not notices.
- A search-discovered MDE bulletin at https://content.govdelivery.com/accounts/MNMDE/bulletins/430cc1a contains school-related **grant** opportunities; it was not substituted for procurement coverage.
- Dependency: supported public listing access without CAPTCHA, or an authenticated/approved collection method outside this task. No challenge bypass attempted.

## North Dakota: public route exists, browser-check barrier

- https://public.ndbuys.nd.gov/ returned HTTP 200 and an official Open Opportunities link.
- Following https://public.ndbuys.nd.gov/page.aspx/en/rfp/request_browse_public in the same session redirected to `/page.aspx/en/bas/browser_check`; HTTP 200 body says the browser is being checked. No real solicitation rows were retrieved.
- Official access instructions: https://www.omb.nd.gov/doing-business-state/bidders/bidding-opportunities-and-resources/suppliers-viewing-and-responding explicitly describe public opportunities without an account; login is therefore not assumed to be inherently required for discovery.
- Dependency: supported Ivalua public browser/session collection that clears its normal browser check. No check bypass or credentials attempted.

## Michigan: denied directory and obsolete guessed portal rejected

- Official https://www.michigan.gov/dtmb/procurement/contractconnect and `/lets-do-business` returned HTTP 403 Access Denied.
- Candidate legacy SIGMA path `https://sigma.michigan.gov/webapp/PRDVSS2X1/AltSelfService` returned HTTP 404 and explicitly said the application was not found. It is not a verified source URL and must not be configured.
- Dependency: recover the current official SIGMA public entry point from an accessible primary page, then verify public record discovery and pagination. No working listing established.

## South Dakota: ESM application shell

- Official procurement guidance points to https://eprocurement.esmsolutions.com/.
- Direct HTTPS fetch returned HTTP 200, title `ESM Purchase`, and no actual solicitation records in the HTML.
- Official reference: https://www.sd.gov/bhra?id=cs_kb_article_view&spa=1&sys_kb_id=a9c554f787aecf14a086bae9cebb356d&sysparm_article=KB0044779
- Dependency: identify and verify the supported public ESM solicitation search/API and its complete state/agency scope. No login requirement asserted without evidence; no collector enabled.
