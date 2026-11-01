# Southeast public-source checks — 2026-10-31

## Louisiana — working official notification subset

`education_southeast.py` collects only current Department-of-Education-funded opportunities from the official Louisiana Board of Regents Contract and RFP Notification System, https://co.apps.laregents.edu/. This is **not complete Louisiana education procurement coverage**.

Live HTTPS verification found three current notification records, one DOE-funded qualifying RFP:

- https://co.apps.laregents.edu/Detail.aspx?id=457
- RFP for Statewide Student Assessments for DOE
- Award Type: RFP; Agency: Office of State Procurement; Respond By: November 10, 2026.
- The source record ID is 457; no solicitation number is stated in the notification, so none is invented.
- Notification links out to the public Ariba discovery event: https://portal.us.bn.cloud.ariba.com/dashboard/public/appext/comsapsbncdiscoveryui#/RfxEvent/preview/1110026412?anId=ANONYMOUS

The adapter binds detail title and funding agency to the CURRENT OPPORTUNITIES listing, checks the complete detail table for cancellation, requires a future explicit Respond By date, and outputs only selected evidence fields. It does not retain award amounts, contact names or addresses. Seven-request maximum; unhandled pagination fails. No attachments or Ariba amendments were fetched, so this is a **review notice**, not assurance that every procurement amendment has been checked.

The public LaPAC education-department route also works:
https://wwwcfprd.doa.louisiana.gov/osp/lapac/dspBid.cfm?search=department&term=14

It currently lists four records, including National Assessment Tutoring and Literacy and Math Tutoring. The list does not establish RFP/RFI type; one identifier explicitly says RFA. Its date column is **Bid Open Date/Time**, not necessarily the submission deadline. No opportunity was fabricated from that directory, and no LaPAC collector was added. The statewide DOE assessment RFP appears under central procurement rather than this department filter.

## Mississippi — department listing blocked

Public primary routing identifies https://mdek12.org/procurement/rfp/ as the RFP/RFQ/RFA/Invitation-to-Bid source. Both this URL and its `www` version returned HTTP 403 from Microsoft Azure Application Gateway with an honest audit user agent. No bypass was attempted.

The separate official no-cost contract page is https://www.mdek12.org/compliance/no-cost-contract-bid-opportunities/. Indexed evidence includes RFQs and intent-to-award notices, so it cannot be substituted as current RFP/RFI coverage. Dependency: permitted public access to the department solicitation listing or an officially published feed with type, status and deadlines.

## Tennessee — central RFP listing blocked

Official state source https://www.arts.tn.gov/partnersforhealth/contracts.html routes active procurements to:
https://www.tn.gov/generalservices/procurement/central-procurement-office--cpo-/supplier-information/request-for-proposals--rfp--opportunities1.html

That current central RFP listing returned HTTP 403. The education school-nutrition procurement page is procedural guidance, not solicitation records. Aggregator descriptions of school-turnaround opportunities were not used as primary evidence or alerts. Dependency: permitted public access to the current central RFP table and its Education records, including amendments and explicit deadlines.

## Georgia — registry requires supported public client

Official DOAS routing: https://www.doas.ga.gov/state-purchasing/bids-and-contracts.
The public registry https://pur.doas.ga.gov/gpr/ returned HTTP 302 to `/gpr/unsupported?browser=` using an honest collector user agent. No browser impersonation or JavaScript bypass was attempted. The department homepage is accessible but did not expose a usable static solicitation listing in this bounded investigation.

Earlier primary indexed record evidence confirms the registry contains typed K–12 agency notices, but a historical record does not prove current collector coverage. Dependency: a supported documented public registry feed/API or permitted public-browser integration with complete pagination and status/agency filters.

## Validation

Six mocked adapter tests pass; Ruff passes. The Louisiana live read returned the single qualifying November 10 review notice. No deployment, credentials, model calls, messages or Mini changes were performed.
