-- @db stocks
CREATE TABLE IF NOT EXISTS valuations (
    ticker TEXT PRIMARY KEY,
    generated_at INTEGER NOT NULL,
    dcf_value REAL,
    eps_value REAL,
    graham_value REAL,
    peg_value REAL,
    fair_value REAL,
    margin_of_safety_pct REAL,
    undervalued INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_valuations_generated_at ON valuations(generated_at DESC);

CREATE TABLE IF NOT EXISTS signals (
    ticker TEXT PRIMARY KEY,
    generated_at INTEGER NOT NULL,
    action TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    reason_code TEXT NOT NULL,
    satisfied_conditions_json TEXT NOT NULL DEFAULT '[]',
    failed_conditions_json TEXT NOT NULL DEFAULT '[]',
    data_warnings_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action, confidence_score DESC);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    ticker TEXT PRIMARY KEY,
    sector TEXT,
    quantity INTEGER NOT NULL,
    avg_cost REAL NOT NULL,
    last_price REAL NOT NULL,
    market_value REAL NOT NULL,
    stop_loss REAL NOT NULL,
    conviction INTEGER NOT NULL DEFAULT 0,
    opened_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_portfolio_sector ON portfolio_positions(sector);

CREATE TABLE IF NOT EXISTS portfolio_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    cash_delta REAL NOT NULL,
    created_at INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_time ON portfolio_transactions(created_at DESC);

-- @db ops
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    artifact_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    active INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(model_name, active, created_at DESC);
