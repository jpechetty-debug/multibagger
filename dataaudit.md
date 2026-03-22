You are a senior quant data engineer auditing the Sovereign AI Trading Engine (Python 3.11, SQLite, NSE Indian stocks, Windows 11).

Build the COMPLETE data audit system as one integrated deliverable. This means all files listed below, fully coded, no placeholders, no TODOs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO BUILD — 4 FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILE 1: engines/audit/data_auditor.py
FILE 2: engines/audit/pre_scan_gate.py
FILE 3: Streamlit "Data Audit" page (add to app/streamlit_app.py)
FILE 4: CLI audit commands (add to sovereign-cli.py)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID RANGES FOR EVERY FIELD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All thresholds must be defined in config.py and referenced from there.
Never hardcode values inside audit logic.

ROE_5Y:           -0.50 to +0.80   | WARN if > 0.60 (suspiciously high)
ROE_TTM:          -1.00 to +1.00   | must directionally match ROE_5Y
SALES_GROWTH_5Y:  -0.30 to +1.00   | WARN if > 0.80 (check base effect)
EPS_GROWTH_TTM:   -2.00 to +5.00   | WARN if > 3.00 (likely base effect from prior loss)
CFO_TO_PAT:        0.00 to  5.00   | WARN if < 0.30 (earnings quality red flag)
DEBT_EQUITY:       0.00 to 10.00   | WARN if > 3.00, FAIL if > 8.00
PEG_RATIO:         0.10 to 10.00   | None is allowed for loss-making stocks
PIOTROSKI_SCORE:   0    to  9      | must be integer, never float
PROMOTER_PCT:      0.0  to 90.0    | WARN if < 20%
PLEDGE_PCT:        0.0  to 100.0   | WARN if > 10%, FAIL if > 30%
PRICE:             > 0             | never None, zero, or negative
MARKET_CAP:        > 1_00_00_000   | 1 Cr minimum — filter shell companies
AVG_VOLUME:        > 10_000        | FAIL if < 10,000 shares daily
DATA_QUALITY_SCORE: 0 to 100       | HARD FAIL if < 50 (from config MIN_DATA_QUALITY)
UPDATED_AT:        < 2 days old    | WARN if stale, HARD FAIL if > 7 days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSS-VALIDATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Price check:       abs(stored_price - live_price) / live_price < 0.02
2. ROE consistency:   abs(roe_5y - roe_ttm) > 0.20 → WARN "ROE_INCONSISTENCY"
3. EPS base effect:   eps_growth_ttm > 3.0 → WARN "EPS_BASE_EFFECT_RISK"
4. CFO quality:       cfo_to_pat < 0.30 → WARN "POOR_EARNINGS_QUALITY"
5. Promoter+pledge:   pledge_pct > 5 AND promoter_pct < 40 → double red flag
6. Pledge source:     if pledge_pct == 0.0 for ALL tickers → source is broken, raise alert
7. PEG recalc:        recalculate PEG = (P/E) / eps_growth, compare vs stored (>10% diff = WARN)
8. Sector check:      sector must be in NSE_SECTOR_LIST defined in config.py
9. Freshness:         updated_at > 2 days → WARN "STALE_DATA"
                      updated_at > 7 days → FAIL "DATA_TOO_OLD" → trigger refetch

SOURCE TRUST HIERARCHY (for disagreements):
1. NSE official API  — highest trust
2. BSE filings       — highest trust for ownership/pledge
3. Alpha Vantage     — medium trust, verify outliers
4. yfinance          — medium trust, verify outliers
5. Calculated values — verify formula implementation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 1: engines/audit/data_auditor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pydantic models:

class FieldAudit(BaseModel):
    field_name: str
    stored_value: Any
    live_value: Any
    status: Literal["PASS", "WARN", "FAIL", "MISSING"]
    reason: str

class AuditReport(BaseModel):
    ticker: str
    timestamp: int
    overall_status: Literal["PASS", "WARN", "FAIL", "INCOMPLETE"]
    data_quality_score: float
    field_results: list[FieldAudit]
    red_flags: list[str]
    suggested_fixes: list[str]

class DataAuditor:

    def audit_ticker(self, ticker: str) -> AuditReport:
        # 1. Load stored data from stocks.db
        # 2. Fetch live data from yfinance as ground truth
        # 3. Run range checks on every field
        # 4. Run all cross-validation rules
        # 5. Calculate data_quality_score (0-100):
        #      score = 100 - (FAIL_count * 20) - (WARN_count * 5)
        # 6. Determine overall_status:
        #      any FAIL → FAIL
        #      any WARN and no FAIL → WARN
        #      all PASS → PASS
        #      >3 MISSING → INCOMPLETE
        # 7. Build suggested_fixes list with exact fix instructions
        # 8. Save AuditReport to ops.db audit_log table
        # 9. Return AuditReport

    def audit_universe(self, tickers: list[str]) -> pd.DataFrame:
        # Run audit_ticker for all tickers
        # Return summary DataFrame: ticker | overall_status | score | red_flags_count | top_red_flag
        # Sort by data_quality_score ascending (worst first)
        # Print FAIL/WARN counts per field name

    def audit_field_distribution(self, field: str) -> dict:
        # Query all values for field from stocks.db
        # Return: min, max, mean, median, p5, p95, p1, p99
        # Flag tickers with values beyond p1/p99 as outliers

    def cross_check_sources(self, ticker: str) -> pd.DataFrame:
        # Fetch from yfinance, nsepython, Alpha Vantage simultaneously
        # Return side-by-side comparison DataFrame per field per source
        # Flag disagreements > 5% difference in red
        # Recommend which source to trust per field

    def fix_data(self, ticker: str, field: str, correct_value: Any):
        # Update stocks.db with corrected value
        # Log to ops.db audit_log: ticker, field, old_value, new_value, timestamp, source="manual"
        # Invalidate cache.db entry for this ticker
        # Print confirmation with before/after

Include __main__ block:
    auditor = DataAuditor()
    for ticker in ["RELIANCE", "TCS", "HDFCBANK"]:
        report = auditor.audit_ticker(ticker)
        print(f"{ticker}: {report.overall_status} | score={report.data_quality_score}")
        for flag in report.red_flags:
            print(f"  RED FLAG: {flag}")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 2: engines/audit/pre_scan_gate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This gate runs inside PipelineOrchestrator._process_ticker()
BEFORE any analysis engine is called.
Bad data must never reach the score engine.

class GateResult(BaseModel):
    passed: bool
    data_quality_score: float
    skip_reason: str | None
    warnings: list[str]

class PreScanGate:

    def check(self, ticker: str, data: FundamentalData) -> GateResult:

        HARD FAILS — skip ticker entirely, log to ops.db:
        - price is None or <= 0                    → "INVALID_PRICE"
        - market_cap < 1_00_00_000                 → "BELOW_MIN_MARKETCAP"
        - data_quality_score < MIN_DATA_QUALITY    → "LOW_DATA_QUALITY"
        - roe_5y is None AND roe_ttm is None        → "NO_ROE_DATA"
        - sales_growth_5y is None                  → "NO_REVENUE_DATA"
        - pledge_pct > 30                          → "EXTREME_PLEDGE"
        - avg_volume < 10_000                      → "ILLIQUID"
        - updated_at > 7 days ago                  → "DATA_TOO_OLD"

        SOFT WARNS — allow but attach to signal output:
        - data_quality_score < 70                  → "LOW_DATA_QUALITY"
        - roe_5y > 0.60                            → "UNUSUALLY_HIGH_ROE"
        - eps_growth_ttm > 3.0                     → "EPS_BASE_EFFECT_RISK"
        - cfo_to_pat < 0.30                        → "POOR_EARNINGS_QUALITY"
        - pledge_pct > 10                          → "ELEVATED_PLEDGE"
        - abs(roe_5y - roe_ttm) > 0.20             → "ROE_INCONSISTENCY"
        - updated_at > 2 days ago                  → "STALE_DATA"
        - peg_ratio is not None and peg_ratio < 0  → "LOSS_MAKING_STOCK"

        Return GateResult with passed=True only if no HARD FAIL.

Integrate into PipelineOrchestrator._process_ticker():
    gate = PreScanGate().check(ticker, data)
    if not gate.passed:
        log_to_ops(ticker, gate.skip_reason, gate.data_quality_score)
        return {"ticker": ticker, "action": "SKIP", "reason": gate.skip_reason}
    result["data_warnings"] = gate.warnings

Add to scan summary output:
    "Skipped 8 tickers: 3 illiquid, 3 low quality, 2 extreme pledge"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 3: STREAMLIT "Data Audit" PAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add page "Data Audit" to app/streamlit_app.py.

Section 1 — Universe health summary
    - Run DataAuditor.audit_universe() with st.cache_data TTL=600
    - 4 metric cards: Total | PASS | WARN | FAIL
    - Plotly bar chart: PASS/WARN/FAIL count per field name
    - Colors: green=PASS, amber=WARN, red=FAIL

Section 2 — Field distribution explorer
    - Selectbox: choose any field
    - Metric cards: min | max | mean | median | p5 | p95
    - Plotly histogram with outliers highlighted red (beyond p1/p99)

Section 3 — Single ticker deep audit
    - Text input: NSE ticker
    - Button: "Run Audit"
    - Table: Field | Stored | Live | Status | Reason
    - Row colors: green/amber/red/gray by status
    - st.error() for each red flag
    - st.info() for each suggested fix

Section 4 — Source cross-check
    - Text input: NSE ticker
    - Button: "Cross-check all sources"
    - Table: Field | yfinance | nsepython | AlphaVantage | Stored | Status
    - Highlight disagreements >5% in red

Section 5 — Manual fix
    - Only show if FAILs exist for selected ticker
    - Selectbox: field to fix
    - Number input: correct value
    - Button: "Apply Fix" → DataAuditor.fix_data() → show before/after

Section 6 — Bulk FAIL report
    - Table of all tickers with overall_status == FAIL
    - Columns: ticker | score | red_flags count | top red flag
    - Button: "Export CSV"
    - Button: "Re-fetch all FAILs" → clears cache + refetches those tickers

Rules:
    - st.cache_data with 10min TTL on expensive operations
    - Graceful empty state: st.info("Run a scan first") if DB empty
    - All DB access via data/db.py — never direct sqlite3 in UI
    - All thresholds from config.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 4: CLI AUDIT COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add these Typer commands to sovereign-cli.py:

1. audit
   python sovereign-cli.py audit --ticker RELIANCE
   python sovereign-cli.py audit --universe QUICK
   python sovereign-cli.py audit --universe STANDARD --export
   → Rich colored table: ticker | score | status | red flags
   → --export saves to data/audit_YYYYMMDD.csv
   → Summary line: "X pass, Y warn, Z fail out of N tickers"

2. verify-field
   python sovereign-cli.py verify-field --field roe_5y
   python sovereign-cli.py verify-field --field pledge_pct --threshold 10
   → Distribution stats for that field across all DB tickers
   → Highlights out-of-range values in red
   → --threshold shows all tickers exceeding that value

3. fix-data
   python sovereign-cli.py fix-data --ticker RELIANCE --field roe_5y --value 0.28
   → Updates stocks.db
   → Logs to ops.db audit_log with before/after
   → Invalidates cache for ticker
   → Prints confirmation

4. cross-check
   python sovereign-cli.py cross-check --ticker RELIANCE
   → Fetches from all sources simultaneously
   → Rich table: Field | yfinance | nsepython | AlphaVantage | Stored | Status
   → Disagreements >5% shown in red
   → Recommends which source to trust per field

5. refetch
   python sovereign-cli.py refetch --ticker RELIANCE
   python sovereign-cli.py refetch --status FAIL
   → Clears cache.db for given ticker(s)
   → Re-fetches from all sources
   → Runs audit after and shows before/after quality score improvement

Use Rich for all output: colored tables, progress bars, summary panels.
Log every command run to ops.db run_history.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE ADDITION — ops.db
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add this table to migrations/002_ops.sql:

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_time    INTEGER DEFAULT (strftime('%s','now')),
    ticker        TEXT,
    field         TEXT,
    old_value     TEXT,
    new_value     TEXT,
    overall_status TEXT,
    data_quality_score REAL,
    red_flags     TEXT,
    source        TEXT,
    action        TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ticker ON audit_log(ticker);
CREATE INDEX IF NOT EXISTS idx_audit_time   ON audit_log(audit_time DESC);

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Complete runnable code — no placeholders, no TODOs
2. Every function has a docstring
3. All DB access via data/db.py context managers only
4. All thresholds and ranges from config.py — never hardcode
5. All data shapes via Pydantic v2 models
6. Every audit action logged to ops.db audit_log table
7. Handle all exceptions with specific messages — no bare except
8. Cache invalidated via data/cache.py whenever data is fixed
9. Every module importable standalone with a __main__ test block
10. Build all 4 files in this order: data_auditor.py → pre_scan_gate.py → streamlit page → CLI commands