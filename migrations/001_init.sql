-- @db stocks
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
    price REAL,
    market_cap REAL,
    avg_volume REAL,
    roe_5y REAL,
    roe_ttm REAL,
    sales_growth_5y REAL,
    eps_growth_ttm REAL,
    cfo_to_pat REAL,
    debt_equity REAL,
    peg_ratio REAL,
    pe_ratio REAL,
    piotroski_score INTEGER,
    promoter_pct REAL,
    pledge_pct REAL,
    fii_delta REAL,
    dii_delta REAL,
    updated_at INTEGER NOT NULL,
    ingestion_quality_score REAL NOT NULL DEFAULT 0,
    ingestion_issues_json TEXT NOT NULL DEFAULT '[]',
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    source_updated_at_json TEXT NOT NULL DEFAULT '{}',
    raw_payload_hash TEXT,
    ingested_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_sector ON fundamentals(sector);
CREATE INDEX IF NOT EXISTS idx_fundamentals_quality ON fundamentals(ingestion_quality_score);
CREATE INDEX IF NOT EXISTS idx_fundamentals_updated_at ON fundamentals(updated_at DESC);

CREATE TABLE IF NOT EXISTS manual_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    field_name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_overrides_ticker_active ON manual_overrides(ticker, active);
CREATE INDEX IF NOT EXISTS idx_overrides_field ON manual_overrides(field_name);

-- @db pit
CREATE TABLE IF NOT EXISTS fundamentals_pit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    captured_at INTEGER NOT NULL,
    fundamentals_json TEXT NOT NULL,
    source_metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_ticker_time ON fundamentals_pit(ticker, captured_at DESC);
