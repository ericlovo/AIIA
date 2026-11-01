# Additional state source checks — 2026-10-31

## Working scoped collectors

### Iowa

Official State of Iowa bid site: https://bidopportunities.iowa.gov/

Its public page contains the Education, Dept Of agency identifier `548843e7-193a-e611-80f0-0050569c6707`. The site's own public JavaScript, https://bidopportunities.iowa.gov/Scripts/Views/Home/HostedBids.js, supplies the documented-in-client JSON endpoint:

https://bidopportunities.iowa.gov/Home/DT_HostedBidsSearch?agencyId=548843e7-193a-e611-80f0-0050569c6707&enteredSearchText=

Live verified TLS: HTTP 200, `iTotalRecords: 0`, `aaData: []` for Education. A separate unfiltered diagnostic returned 46 actual solicitation records with agency, Open status, effective and expiration dates; the endpoint returns its complete filtered collection rather than honoring DataTables page-length parameters. The collector requires array length to equal the explicit total and caps records and response bytes, failing on partial results. Zero Education records is a verified scoped no-match result, not a claim of all Iowa school procurement coverage.

`education_additional.py` validates agency ID and agency name, Open status, active boolean, solicitation type, K–12 context and future explicit expiration date. It converts the site's .NET timestamp into America/Chicago dates. Contact fields and free-text descriptions are not persisted; only the selected title, notice number, buyer and deadline are emitted. Detail links are stable `Home/BidInfo?bidId=<uuid>` URLs.

### Indiana

Working current source: https://www.in.gov/idoa/procurement/current-business-opportunities/

The old `/index.html` URL returns HTTP 404. The current trailing-slash route returns HTTP 200 with the full server-rendered `events-table`, whose columns are Event Name, Agency, Event ID, Event Description, Response Due By and Contact. Display pagination operates over the embedded rows. The collector requires this exact table/header schema and fails on any server-pagination link.

Live result: zero qualifying Department-of-Education RFP/RFI rows. The scope is the state education agency, not all districts. Due dates come from Response Due By, never from conference or question dates. Event descriptions are inspected for type and cancellation, but contact and description fields are excluded from emitted evidence.

## Remaining sources: exact dependencies, not claimed coverage

### Alaska

https://aws.state.ak.us/OnlinePublicNotices/ returns HTTP 200 with actual recent statewide notices. Its advertised active-browse route https://aws.state.ak.us/OnlinePublicNotices/Notices/Browse.aspx returned HTTP 403. The homepage's recent notices are incomplete and cannot support statewide or education-agency completeness. Dependency: permitted public browse/search or an official feed with agency filter, deadlines and complete pagination. No collector added.

### Arizona

Official ADE procurement page https://cms.azed.gov/adeinfo/about-ade/about-department/accounting-procurement returns HTTP 403 with a Cloudflare JavaScript/cookie challenge. Earlier primary routing directs ADE solicitations to the Arizona Procurement Portal. Dependency: a permitted public APP feed/search adapter with actual agency-filtered records. No challenge bypass and no collector added.

### Arkansas

https://arbuy.arkansas.gov/bso/ returns HTTP 200 and explicitly states that the state migrated to SAP Ariba in July 2026; ARBuy retains only solicitations opened before migration. Current official routing is https://sas.arkansas.gov/procurement. A legacy ARBuy collector would understate current coverage. Dependency: verified public Ariba current-solici­tation interface and education-agency filter, with legacy notices handled separately. No collector added.

### Colorado

https://www.bidscolorado.com/co/portal.nsf/xpRegistration.xsp returns HTTP 200 but states that registration is retired and most solicitations moved to ColoradoVSS. It links the state VSS route and state purchasing information; it is not an actual current notice feed. Dependency: current public ColoradoVSS records and agency/status/date/pagination adapter. No collector added.

### Florida

https://www.fldoe.org/finance/contracts-grants-procurement/purchasing/ returned HTTP 403 Akamai Access Denied with an honest collector user agent. Dependency: permitted department listing or official central MyFloridaMarketPlace education-agency feed. No secondary aggregator opportunity was promoted to an official alert. No collector added.

### Kentucky

https://www.education.ky.gov/districts/business/Pages/default.aspx returns HTTP 200, but is a routing/guidance page. Its published procurement destinations are https://finance.ky.gov/eProcurement/Pages/default.aspx and https://vss.ky.gov/vssprod-ext/Advantage4. Personal tracking parameters in the page's wrapped links are neither needed nor retained. Dependency: bounded public VSS search/API with education-agency, type, status and pagination verification. No collector added.

## Verification

Seven mocked tests and Ruff pass for `education_additional.py`. Live read-only TLS probes verify Iowa and Indiana scoped no-match outcomes. No credentials, model calls, messaging, deployments, Mini changes, browser impersonation, challenge bypasses or TLS exceptions were used.
