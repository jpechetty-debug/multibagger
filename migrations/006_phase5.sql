-- @db stocks
CREATE TABLE IF NOT EXISTS analysis_snapshots (
    ticker TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    as_of INTEGER NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (ticker, analysis_type)
);
CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_type ON analysis_snapshots(analysis_type, as_of DESC);

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_type TEXT PRIMARY KEY,
    as_of INTEGER NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_as_of ON market_snapshots(as_of DESC);

CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    generated_at INTEGER NOT NULL,
    regime TEXT NOT NULL,
    weighted_score REAL NOT NULL,
    meta_model_score REAL NOT NULL,
    total_score REAL NOT NULL,
    action TEXT NOT NULL,
    factor_scores_json TEXT NOT NULL DEFAULT '[]',
    feature_names_json TEXT NOT NULL DEFAULT '[]',
    feature_values_json TEXT NOT NULL DEFAULT '[]',
    reasoning_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_score_history_ticker_time ON score_history(ticker, generated_at DESC);
