#!/bin/bash
# AIIA Security Scan — local vulnerability & secret scanning suite
# Runs: trufflehog, trivy, bandit, semgrep, shellcheck, hadolint, pip-audit
#
# Usage:
#   scripts/security_scan.sh           # full suite
#   scripts/security_scan.sh --quick   # secrets + deps only (trufflehog + trivy)
#
# Output:
#   ./security-reports/<date>/          # per-scanner JSON + baseline_filtered.json
#   ./security-reports/latest.txt       # human summary
#   ./security-reports/latest-report    # symlink to most recent dated dir
#
# Baseline filter:
#   Findings listed in .security-baseline.json are suppressed. See that file
#   for the schema. filter_security_baseline.py runs after all scanners.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${AIIA_SECURITY_LOG_DIR:-$REPO_DIR/security-reports}"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
REPORT_DIR="$LOG_DIR/$DATE"
SUMMARY="$LOG_DIR/latest.txt"
QUICK=false

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick|-q) QUICK=true; shift ;;
        *) shift ;;
    esac
done

mkdir -p "$REPORT_DIR"

header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  AIIA Security Scan — $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
    echo ""
}

section() { echo -e "\n${BOLD}[$1]${NC} $2"; }
pass()    { echo -e "  ${GREEN}PASS${NC} $1"; }
warn()    { echo -e "  ${YELLOW}WARN${NC} $1"; }
fail()    { echo -e "  ${RED}FAIL${NC} $1"; }
skip()    { echo -e "  ${YELLOW}SKIP${NC} $1"; }

R_TRUFFLEHOG="N/A"
R_TRIVY="N/A"
R_BANDIT="N/A"
R_SEMGREP="N/A"
R_SHELLCHECK="N/A"
R_HADOLINT="N/A"
R_PIPAUDIT="N/A"
OVERALL_EXIT=0

cd "$REPO_DIR"

if [ "$QUICK" = true ]; then
    header "Quick Scan ($DATE)"
else
    header "Full Suite ($DATE)"
fi

echo "  Repo:    $REPO_DIR"
echo "  Reports: $REPORT_DIR"
echo "  Time:    $TIMESTAMP"

# ===========================================================================
# 1. TRUFFLEHOG — Secret Detection
# ===========================================================================
section "1/7" "trufflehog — Secret Detection"

if command -v trufflehog &>/dev/null; then
    EXCLUDE_FILE="$REPO_DIR/.trufflehog-exclude"
    EXCLUDE_PATHS=$(echo -e "node_modules\n.git\nvenv\n.venv\n__pycache__\nsecurity-reports")
    if [ -f "$EXCLUDE_FILE" ]; then
        FILE_EXCLUDES=$(grep -v '^\s*#' "$EXCLUDE_FILE" | grep -v '^\s*$' || true)
        if [ -n "$FILE_EXCLUDES" ]; then
            EXCLUDE_PATHS="$EXCLUDE_PATHS"$'\n'"$FILE_EXCLUDES"
        fi
    fi

    trufflehog filesystem "$REPO_DIR" \
        --exclude-paths <(echo "$EXCLUDE_PATHS") \
        --json \
        --no-update \
        > "$REPORT_DIR/trufflehog.json" 2>/dev/null

    SECRET_COUNT=$(wc -l < "$REPORT_DIR/trufflehog.json" | tr -d ' ')
    if [ "$SECRET_COUNT" -eq 0 ]; then
        pass "No secrets detected"
        R_TRUFFLEHOG="PASS (0 findings)"
    else
        fail "$SECRET_COUNT potential secret(s) found"
        R_TRUFFLEHOG="FAIL ($SECRET_COUNT findings)"
        OVERALL_EXIT=1
    fi
else
    skip "trufflehog not installed"
    R_TRUFFLEHOG="SKIP"
fi

# ===========================================================================
# 2. TRIVY — Dependency Vulnerability Scan
# ===========================================================================
section "2/7" "trivy — Dependency Vulnerabilities"

if command -v trivy &>/dev/null; then
    trivy fs "$REPO_DIR" \
        --scanners vuln \
        --severity CRITICAL,HIGH \
        --skip-dirs node_modules,.git,venv,.venv,security-reports \
        --format json \
        --output "$REPORT_DIR/trivy-deps.json" \
        --quiet 2>/dev/null

    CRITICAL=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$REPORT_DIR/trivy-deps.json" 2>/dev/null || echo "0")
    HIGH=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$REPORT_DIR/trivy-deps.json" 2>/dev/null || echo "0")

    if [ "$CRITICAL" -gt 0 ]; then
        fail "$CRITICAL critical, $HIGH high vulnerabilities"
        R_TRIVY="FAIL ($CRITICAL critical, $HIGH high)"
        OVERALL_EXIT=1
    elif [ "$HIGH" -gt 0 ]; then
        warn "$HIGH high vulnerabilities (0 critical)"
        R_TRIVY="WARN ($HIGH high)"
    else
        pass "No critical/high vulnerabilities"
        R_TRIVY="PASS"
    fi
else
    skip "trivy not installed"
    R_TRIVY="SKIP"
fi

if [ "$QUICK" = true ]; then
    echo ""
    echo -e "${BOLD}Quick scan complete.${NC} Use full scan for SAST + pattern matching."
    cat > "$SUMMARY" << EOF
AIIA Security Scan — Quick ($DATE)
===================================
Scan Time:  $TIMESTAMP
Mode:       Quick (secrets + deps)

Results:
  trufflehog:  $R_TRUFFLEHOG
  trivy:       $R_TRIVY

Reports: $REPORT_DIR
EOF
    cat "$SUMMARY"
    ln -sfn "$REPORT_DIR" "$LOG_DIR/latest-report"
    exit $OVERALL_EXIT
fi

# ===========================================================================
# 3. BANDIT — Python SAST
# ===========================================================================
section "3/7" "bandit — Python Static Analysis (SAST)"

if command -v bandit &>/dev/null; then
    bandit -r local_brain \
        -f json -o "$REPORT_DIR/bandit.json" \
        -ll \
        --exclude '*/tests/*,*/__pycache__/*,*/.venv/*,*/venv/*' \
        2>/dev/null || true

    if [ -f "$REPORT_DIR/bandit.json" ]; then
        B_HIGH=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' "$REPORT_DIR/bandit.json" 2>/dev/null || echo "0")
        B_MED=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' "$REPORT_DIR/bandit.json" 2>/dev/null || echo "0")
        B_LOW=$(jq '[.results[] | select(.issue_severity == "LOW")] | length' "$REPORT_DIR/bandit.json" 2>/dev/null || echo "0")

        if [ "$B_HIGH" -gt 0 ]; then
            fail "$B_HIGH high, $B_MED medium, $B_LOW low"
            R_BANDIT="FAIL ($B_HIGH high, $B_MED medium)"
            OVERALL_EXIT=1
        elif [ "$B_MED" -gt 0 ]; then
            warn "$B_MED medium, $B_LOW low (0 high)"
            R_BANDIT="WARN ($B_MED medium)"
        else
            pass "No high/medium findings ($B_LOW low)"
            R_BANDIT="PASS ($B_LOW low)"
        fi
    else
        warn "Report not generated"
        R_BANDIT="WARN (no report)"
    fi
else
    skip "bandit not installed"
    R_BANDIT="SKIP"
fi

# ===========================================================================
# 4. SEMGREP — Pattern-Based Security Analysis
# ===========================================================================
section "4/7" "semgrep — Pattern Analysis"

if command -v semgrep &>/dev/null; then
    semgrep scan \
        --config auto \
        --severity ERROR \
        --json \
        --output "$REPORT_DIR/semgrep.json" \
        --exclude "node_modules" \
        --exclude ".git" \
        --exclude "venv" \
        --exclude ".venv" \
        --exclude "security-reports" \
        --quiet \
        local_brain \
        2>/dev/null || true

    if [ -f "$REPORT_DIR/semgrep.json" ]; then
        SG_COUNT=$(jq '.results | length' "$REPORT_DIR/semgrep.json" 2>/dev/null || echo "0")
        SG_ERRORS=$(jq '[.results[] | select(.extra.severity == "ERROR")] | length' "$REPORT_DIR/semgrep.json" 2>/dev/null || echo "0")

        if [ "$SG_ERRORS" -gt 0 ]; then
            fail "$SG_ERRORS error-level findings"
            R_SEMGREP="FAIL ($SG_ERRORS errors)"
            OVERALL_EXIT=1
        elif [ "$SG_COUNT" -gt 0 ]; then
            warn "$SG_COUNT findings (no errors)"
            R_SEMGREP="WARN ($SG_COUNT findings)"
        else
            pass "No findings"
            R_SEMGREP="PASS"
        fi
    else
        warn "Report not generated"
        R_SEMGREP="WARN (no report)"
    fi
else
    skip "semgrep not installed"
    R_SEMGREP="SKIP"
fi

# ===========================================================================
# 5. SHELLCHECK — Shell Script Analysis
# ===========================================================================
section "5/7" "shellcheck — Shell Script Analysis"

if command -v shellcheck &>/dev/null; then
    SC_ERRORS=0
    SC_WARNINGS=0
    SC_REPORT="$REPORT_DIR/shellcheck.json"
    echo "[]" > "$SC_REPORT"

    while IFS= read -r f; do
        [ -z "$f" ] && continue
        result=$(shellcheck -f json "$f" 2>/dev/null || true)
        if [ -n "$result" ] && [ "$result" != "[]" ]; then
            errors=$(echo "$result" | jq '[.[] | select(.level == "error")] | length' 2>/dev/null || echo "0")
            warnings=$(echo "$result" | jq '[.[] | select(.level == "warning")] | length' 2>/dev/null || echo "0")
            SC_ERRORS=$((SC_ERRORS + errors))
            SC_WARNINGS=$((SC_WARNINGS + warnings))
            jq -s '.[0] + .[1]' "$SC_REPORT" <(echo "$result") > "$SC_REPORT.tmp" && mv "$SC_REPORT.tmp" "$SC_REPORT"
        fi
    done < <(find "$REPO_DIR" -type f -name "*.sh" \
        -not -path "*/node_modules/*" \
        -not -path "*/.git/*" \
        -not -path "*/.venv/*" \
        -not -path "*/venv/*" \
        -not -path "*/security-reports/*" 2>/dev/null)

    if [ "$SC_ERRORS" -gt 0 ]; then
        fail "$SC_ERRORS errors, $SC_WARNINGS warnings"
        R_SHELLCHECK="FAIL ($SC_ERRORS errors)"
        OVERALL_EXIT=1
    elif [ "$SC_WARNINGS" -gt 0 ]; then
        warn "$SC_WARNINGS warnings (0 errors)"
        R_SHELLCHECK="WARN ($SC_WARNINGS warnings)"
    else
        pass "All shell scripts clean"
        R_SHELLCHECK="PASS"
    fi
else
    skip "shellcheck not installed"
    R_SHELLCHECK="SKIP"
fi

# ===========================================================================
# 6. HADOLINT — Dockerfile Analysis
# ===========================================================================
section "6/7" "hadolint — Dockerfile Analysis"

if command -v hadolint &>/dev/null; then
    HL_ERRORS=0
    HL_WARNINGS=0
    HL_REPORT="$REPORT_DIR/hadolint.json"
    echo "[]" > "$HL_REPORT"

    while IFS= read -r df; do
        [ -z "$df" ] && continue
        result=$(hadolint -f json "$df" 2>/dev/null || true)
        if [ -n "$result" ] && [ "$result" != "[]" ]; then
            errors=$(echo "$result" | jq '[.[] | select(.level == "error")] | length' 2>/dev/null || echo "0")
            warnings=$(echo "$result" | jq '[.[] | select(.level == "warning")] | length' 2>/dev/null || echo "0")
            HL_ERRORS=$((HL_ERRORS + errors))
            HL_WARNINGS=$((HL_WARNINGS + warnings))
            jq -s '.[0] + .[1]' "$HL_REPORT" <(echo "$result") > "$HL_REPORT.tmp" && mv "$HL_REPORT.tmp" "$HL_REPORT"
        fi
    done < <(find "$REPO_DIR" -type f -name "Dockerfile*" \
        -not -path "*/node_modules/*" \
        -not -path "*/.git/*" \
        -not -path "*/security-reports/*" 2>/dev/null)

    if [ "$HL_ERRORS" -gt 0 ]; then
        fail "$HL_ERRORS errors, $HL_WARNINGS warnings"
        R_HADOLINT="FAIL ($HL_ERRORS errors)"
        OVERALL_EXIT=1
    elif [ "$HL_WARNINGS" -gt 0 ]; then
        warn "$HL_WARNINGS warnings (0 errors)"
        R_HADOLINT="WARN ($HL_WARNINGS warnings)"
    else
        pass "All Dockerfiles clean"
        R_HADOLINT="PASS"
    fi
else
    skip "hadolint not installed"
    R_HADOLINT="SKIP"
fi

# ===========================================================================
# 7. PIP-AUDIT — Python Dependency CVE Scan
# ===========================================================================
section "7/7" "pip-audit — Python Dependency CVEs"

if command -v pip-audit &>/dev/null; then
    PA_TOTAL=0
    PA_REPORT="$REPORT_DIR/pip-audit.json"
    echo "[]" > "$PA_REPORT"

    if [ -f "$REPO_DIR/requirements.txt" ]; then
        result=$(pip-audit -r "$REPO_DIR/requirements.txt" -f json --no-deps 2>/dev/null || true)
        if [ -n "$result" ]; then
            count=$(echo "$result" | jq '.dependencies | [.[] | select(.vulns | length > 0)] | length' 2>/dev/null || echo "0")
            PA_TOTAL=$((PA_TOTAL + count))
            echo "$result" > "$PA_REPORT" 2>/dev/null || true
        fi
    fi

    if [ "$PA_TOTAL" -gt 0 ]; then
        fail "$PA_TOTAL package(s) with known vulnerabilities"
        R_PIPAUDIT="FAIL ($PA_TOTAL vulnerable)"
        OVERALL_EXIT=1
    else
        pass "No known vulnerabilities in pinned dependencies"
        R_PIPAUDIT="PASS"
    fi
else
    skip "pip-audit not installed (install: pipx install pip-audit)"
    R_PIPAUDIT="SKIP"
fi

# ===========================================================================
# BASELINE FILTER
# ===========================================================================
section "POST" "Baseline filter — suppressing accepted findings"

BASELINE_STATUS=""
if [ -f "$REPO_DIR/.security-baseline.json" ] && command -v python3 &>/dev/null; then
    BASELINE_OUTPUT=$(REPO_DIR="$REPO_DIR" python3 "$REPO_DIR/scripts/filter_security_baseline.py" "$REPORT_DIR" "$SUMMARY" 2>/dev/null) || true
    BASELINE_NEW=$(echo "$BASELINE_OUTPUT" | grep "BASELINE_NEW=" | cut -d= -f2)
    BASELINE_ACCEPTED=$(echo "$BASELINE_OUTPUT" | grep "BASELINE_ACCEPTED=" | cut -d= -f2)

    if [ -n "$BASELINE_NEW" ] && [ "$BASELINE_NEW" -eq 0 ] 2>/dev/null; then
        pass "All findings baselined ($BASELINE_ACCEPTED accepted)"
        BASELINE_STATUS=" [effective: PASS — $BASELINE_ACCEPTED baselined]"
        OVERALL_EXIT=0
    elif [ -n "$BASELINE_NEW" ]; then
        warn "$BASELINE_NEW new finding(s) not in baseline"
        BASELINE_STATUS=" [new: $BASELINE_NEW, baselined: ${BASELINE_ACCEPTED:-0}]"
    fi

    BF_JSON="$REPORT_DIR/baseline_filtered.json"
    if [ -f "$BF_JSON" ]; then
        for scanner in trufflehog bandit semgrep shellcheck hadolint; do
            new=$(jq -r ".${scanner}.new // 0" "$BF_JSON" 2>/dev/null)
            baselined=$(jq -r ".${scanner}.baselined // 0" "$BF_JSON" 2>/dev/null)
            total=$(jq -r ".${scanner}.total // 0" "$BF_JSON" 2>/dev/null)
            if [ "$new" = "0" ] && [ "$total" != "0" ]; then
                case "$scanner" in
                    trufflehog) R_TRUFFLEHOG="PASS ($baselined baselined)" ;;
                    bandit)     R_BANDIT="PASS ($baselined baselined)" ;;
                    semgrep)    R_SEMGREP="PASS ($baselined baselined)" ;;
                    shellcheck) R_SHELLCHECK="PASS ($baselined baselined)" ;;
                    hadolint)   R_HADOLINT="PASS ($baselined baselined)" ;;
                esac
            fi
        done
    fi
else
    skip "No baseline file or python3 not available"
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════${NC}"
echo ""

cat > "$SUMMARY" << EOF
AIIA Security Scan — Full Suite ($DATE)
========================================
Scan Time:  $TIMESTAMP
Mode:       Full (all 7 scanners)
Repo:       $REPO_DIR

Results:
  1. trufflehog (secrets):    $R_TRUFFLEHOG
  2. trivy (deps/vulns):      $R_TRIVY
  3. bandit (Python SAST):    $R_BANDIT
  4. semgrep (patterns):      $R_SEMGREP
  5. shellcheck (scripts):    $R_SHELLCHECK
  6. hadolint (Dockerfiles):  $R_HADOLINT
  7. pip-audit (Python CVEs): $R_PIPAUDIT

Overall: $([ $OVERALL_EXIT -eq 0 ] && echo "PASS" || echo "FAIL")${BASELINE_STATUS}

Reports: $REPORT_DIR
EOF

cat "$SUMMARY"

if [ -f "$REPORT_DIR/baseline_filtered.json" ]; then
    echo ""
    echo "Baseline details: $REPORT_DIR/baseline_filtered.json"
fi

ln -sfn "$REPORT_DIR" "$LOG_DIR/latest-report"

# Cleanup old reports (keep 30 days)
find "$LOG_DIR" -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} \; 2>/dev/null || true

exit $OVERALL_EXIT
