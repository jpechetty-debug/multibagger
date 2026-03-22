-- @db ops
CREATE TABLE IF NOT EXISTS audit_universe_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_id TEXT NOT NULL UNIQUE,
    audit_time INTEGER NOT NULL,
    triggered_by TEXT NOT NULL,
    tickers_json TEXT NOT NULL DEFAULT '[]',
    tickers_audited INTEGER NOT NULL,
    pass_count INTEGER NOT NULL,
    warn_count INTEGER NOT NULL,
    fail_count INTEGER NOT NULL,
    incomplete_count INTEGER NOT NULL,
    average_score REAL,
    median_score REAL,
    field_fail_counts_json TEXT NOT NULL DEFAULT '{}',
    field_warn_counts_json TEXT NOT NULL DEFAULT '{}',
    field_missing_counts_json TEXT NOT NULL DEFAULT '{}',
    score_distribution_json TEXT NOT NULL DEFAULT '{}',
    source_health_alerts_json TEXT NOT NULL DEFAULT '[]',
    report_rows_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_audit_universe_runs_time ON audit_universe_runs(audit_time DESC);
