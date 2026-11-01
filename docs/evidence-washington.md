# Washington OSPI via official WEBS

Verified 2026-10-31 using normal TLS verification and no proxy workaround or authentication.

The [public WEBS calendar](https://pr-webs-vendor.des.wa.gov/BidCalendar.aspx) exposes an ASP.NET organization selector. The adapter retrieves the page, selects the exact published agency label `Superintendent of Public Instruction, Office of`, then posts the server-provided hidden form state and Search action to the same URL. The selected value is discovered from the official form rather than guessed. Returned selection must confirm the same agency.

Actual two-request adapter execution returned one qualifying notice:

- [Processed Foods Using USDA Commodities](https://pr-webs-vendor.des.wa.gov/Search_BidDetails.aspx?ID=57699)
- WEBS reference: RFP 2026-07.
- Solicitation close date: December 3, 2026. The separate question deadline is November 10; the parser does not confuse that date with closure.
- Published description identifies OSPI as purchaser distributing products to local education agencies.

This is OSPI coverage only, not every Washington school district or every WEBS agency. The original OSPI website access failure is not bypassed. No login is attempted. Third-column contact names are excluded from returned evidence.

The current filtered result has one page. If paging controls appear, the adapter raises an incomplete-source error instead of presenting its first page as complete. Missing form state, agency selection, grid structure, redirects, oversized responses and malformed deadlines also fail. Same-day and expired dates are withheld; unknown cutoff times require checking the official notice. Only explicit RFP/RFI references qualify. Cancellation, withdrawal, closed and award language causes withholding.

Interface: `fetch(client, source, today)` with source URL exactly the public calendar above and state WA or Washington. No registry, worker, model, Slack or schedule activation is included in these changes.
