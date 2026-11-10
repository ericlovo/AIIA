# Private BD ICP profiles

Segment bots can load a **private** ideal-customer profile and a do-not-prospect
suppression list from outside git. The first user is the `healthcare` bot.
Missing files leave discovery, extraction, judgment, and candidate writes
unchanged. Malformed files fail closed for that bot: they are unused, logged
without contents, and `icp_status` on `/api/bd/snapshot` is `error`. The
conductor does not crash.

These files are **not** in the repository. Do not commit real customer, company,
or deal names. The examples below are synthetic.

## Location

Default directory (mode `0700` on the Mini):

```text
~/.aiia-private/bd-icp/<bot_id>/*.icp.json
~/.aiia-private/bd-icp/<bot_id>/*.suppression.json
```

Override with environment variable `AIIA_BD_ICP_DIR` or an optional roster
config key `icp_dir`. Precedence: `AIIA_BD_ICP_DIR`, then `icp_dir`, then the
default path. Only files whose `bot_ids` include the bot are loaded.

## File shapes (version 1)

`*.suppression.json`:

```json
{
  "version": 1,
  "customer": "Acme Widgets",
  "bot_ids": ["healthcare"],
  "reason": "existing customer pipeline",
  "updated": "2026-11-10",
  "organizations": ["Acme Widgets, Inc.", "Example County Hospital"],
  "notes": "Synthetic fixture; never commit real names"
}
```

`*.icp.json`:

```json
{
  "version": 1,
  "customer": "Acme Widgets",
  "bot_ids": ["healthcare"],
  "updated": "2026-11-10",
  "source": "operator",
  "buyer_types": {
    "hot": ["health systems", "community hospitals", "county providers"],
    "downweight": ["retail clinics"]
  },
  "employer_sectors": ["healthcare providers"],
  "employer_size": "mid-market",
  "geography": {
    "priority_1": ["Wisconsin", "Example County"],
    "priority_2": ["Midwest"],
    "lower": ["West Coast"]
  },
  "products": {
    "BENEFITS": {
      "name": "Benefits administration",
      "value_share": 0.4,
      "typical_ratio_of_sits": [0.143, 0.175]
    }
  },
  "channels": {
    "treat_as": "channel/partner, not end buyer",
    "known": ["brokers", "co-ops", "health plans"]
  },
  "signals": ["expansion", "hiring", "new clinic"],
  "stages_note": "Prefer expansion and hiring over brand mentions."
}
```

`typical_ratio_of_sits` may be a single number in `[0, 1]` or a list of one or
two such numbers (a range). The compact summary renders a range as
`14-18% of SITS`. Bools, strings, nested lists, and values outside `[0, 1]`
fail closed. Malformed files log the failing **top-level key name only**.

## Behavior

- **Suppression** is a hard gate after Qwen extraction and before Jev or any
  candidate write. Organization names are normalized (case, punctuation,
  Inc/LLC/Co/Ltd suffixes, `&`/`and`, parentheticals, trailing ` - place`
  suffixes) and matched exactly or with a conservative token-overlap/fuzzy
  check. A match is recorded as withheld with reason
  `suppressed_existing_customer_pipeline` and does not call Jev. During the
  eight-item Qwen pre-screen, suppressed orgs are skipped so another item can
  be sent to judgment.
- **Targeting** shapes the Google News RSS query toward hot buyer types,
  employer sectors, and `priority_1` / `priority_2` geography, and appends a
  compact ICP summary (buyer types, downweight lanes, geography, channels as
  channels/partners, product emphasis) to the Qwen extraction prompt and the
  Jev public state. The summary stays well under ~600 tokens and never includes
  the customer name or the suppression list. Brokers, co-ops, and health plans
  are labeled channel/partner, not end buyer.
- **Snapshot** `/api/bd/snapshot` per bot includes `icp_loaded`, the ICP
  customer label, `suppression_count` (a number only), `icp_status`, and
  `suppressed_withholds_today`. It never lists suppression organization names.

Withhold thresholds (0.8), budgets, cadence, kill switch, review gates, and
airgap policy are unchanged.
