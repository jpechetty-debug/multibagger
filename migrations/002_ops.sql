-- @db ops
CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_name TEXT NOT NULL,
    command_args_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    summary TEXT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_history_time ON run_history(finished_at DESC);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_logs_component_time ON logs(component, created_at DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_time INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    ticker TEXT,
    field TEXT,
    old_value TEXT,
    new_value TEXT,
    overall_status TEXT,
    data_quality_score REAL,
    red_flags TEXT,
    source TEXT,
    action TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ticker ON audit_log(ticker);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(audit_time DESC);

CREATE TABLE IF NOT EXISTS audit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    audit_time INTEGER NOT NULL,
    overall_status TEXT NOT NULL,
    audit_quality_score REAL NOT NULL,
    fail_count INTEGER NOT NULL,
    warn_count INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    red_flags_json TEXT NOT NULL DEFAULT '[]',
    suggested_fixes_json TEXT NOT NULL DEFAULT '[]',
    triggered_by TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_runs_ticker_time ON audit_runs(ticker, audit_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_runs_status ON audit_runs(overall_status, audit_time DESC);

CREATE TABLE IF NOT EXISTS audit_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    finding_code TEXT NOT NULL,
    field_name TEXT NOT NULL,
    stored_value_json TEXT,
    resolved_live_value_json TEXT,
    source_name TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    numeric_delta REAL
);
CREATE INDEX IF NOT EXISTS idx_audit_findings_run ON audit_findings(run_id);
CREATE INDEX IF NOT EXISTS idx_audit_findings_field ON audit_findings(field_name, status);

CREATE TABLE IF NOT EXISTS audit_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    field_name TEXT,
    old_value_json TEXT,
    new_value_json TEXT,
    action_type TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    related_run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_actions_ticker_time ON audit_actions(ticker, created_at DESC);
