# Security Scanning

AIIA ships a local security scan suite that runs seven industry-standard
scanners against the repo and produces a single pass/fail summary. Findings
that have been reviewed and accepted are suppressed via a baseline file, so
the summary stays signal-only.

## Quick start

```bash
# Full suite
scripts/security_scan.sh

# Secrets + dependencies only (fast)
scripts/security_scan.sh --quick
```

Reports are written to `./security-reports/<date>/`, with a human summary
at `./security-reports/latest.txt` and a `latest-report` symlink pointing at
the most recent dated directory.

Override the output location by exporting `AIIA_SECURITY_LOG_DIR` before
running the script.

## Scanners

| # | Tool         | Purpose                         | Install                      |
|---|--------------|---------------------------------|------------------------------|
| 1 | trufflehog   | Secret detection                | `brew install trufflehog`    |
| 2 | trivy        | Dependency CVEs                 | `brew install trivy`         |
| 3 | bandit       | Python SAST                     | `pip install bandit`         |
| 4 | semgrep      | Pattern-based static analysis   | `pip install semgrep`        |
| 5 | shellcheck   | Shell script linting            | `brew install shellcheck`    |
| 6 | hadolint     | Dockerfile linting              | `brew install hadolint`      |
| 7 | pip-audit    | Python dep CVEs (OSV database)  | `pipx install pip-audit`     |

Any scanner not installed is marked `SKIP` and does not fail the run. The
suite is designed to be resilient to missing tools so you can start with
the ones you have and add more over time.

## Baseline filter

Every medium-sized codebase accumulates some findings that are genuinely
safe — a `0.0.0.0` bind required for container networking, a hardcoded
connection string in `.env.example`, a `B102 exec` inside a sandboxed
evaluator. Failing the build on these teaches the team to ignore the
scanner, which is the worst possible outcome.

`.security-baseline.json` is a list of accepted findings. After all
scanners run, `scripts/filter_security_baseline.py` diffs the raw results
against the baseline and computes **new** findings. The overall exit code
reflects only new findings — a run with 20 baselined hits and 0 new
findings is a PASS.

### Schema

```json
{
  "description": "Accepted security findings...",
  "updated": "2026-04-12",
  "accepted": [
    {
      "scanner": "bandit",
      "rule": "B104",
      "file_pattern": "local_api.py",
      "reason": "0.0.0.0 bind required for docker container networking"
    }
  ]
}
```

| Field          | Required | Description                                               |
|----------------|----------|-----------------------------------------------------------|
| `scanner`      | yes      | `trufflehog`, `bandit`, `semgrep`, `shellcheck`, `hadolint` |
| `rule`         | yes      | Rule ID (e.g. `B104`, `DL3008`, `avoid-sqlalchemy-text`)   |
| `file_pattern` | no       | Substring match against finding path. Empty = match all. |
| `reason`       | yes      | Why this finding is accepted. Required for future-you.   |

`file_pattern` uses substring matching, not glob. `local_brain/agentfs.py`
matches both the exact path and any finding under that directory.

### Adding a finding to the baseline

1. Run the scanner and inspect the finding in
   `./security-reports/<date>/<scanner>.json`.
2. Decide whether it's a real issue or a safe-by-design pattern.
   - If real → fix the code.
   - If safe → add an entry to `.security-baseline.json` with a clear
     `reason` that explains *why* it's safe.
3. Re-run `scripts/security_scan.sh` and confirm the finding is now
   counted as `baselined` instead of `new`.

## CI integration

The scan script exits non-zero on new findings, so it plugs straight into
CI:

```yaml
- name: Security scan
  run: scripts/security_scan.sh
```

For a faster pre-merge gate, use `--quick` to only run trufflehog and
trivy.

## Nightly scans

On a workstation, you can run the full suite nightly via launchd (macOS)
or cron (Linux). The script writes to `./security-reports/<date>/` and
cleans up reports older than 30 days automatically.

Example launchd plist — runs midnight local time:

```xml
<plist version="1.0">
<dict>
  <key>Label</key><string>org.aiia.securityscan</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>cd /path/to/AIIA &amp;&amp; scripts/security_scan.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>0</integer><key>Minute</key><integer>0</integer></dict>
</dict>
</plist>
```
