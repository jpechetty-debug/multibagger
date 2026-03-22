-- @db swing
CREATE TABLE IF NOT EXISTS swing_signals (
    ticker TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    price REAL,
    rsi REAL,
    trend INTEGER,
    volume_surge INTEGER,
    target REAL,
    stop_loss REAL,
    generated_at INTEGER NOT NULL
);

-- @db mb
CREATE TABLE IF NOT EXISTS multibagger_candidates (
    ticker TEXT PRIMARY KEY,
    conviction_score REAL NOT NULL,
    quality_score REAL NOT NULL,
    early_signal_score REAL NOT NULL,
    tam_score REAL NOT NULL,
    price REAL,
    action TEXT NOT NULL,
    tranche_plan_json TEXT,
    generated_at INTEGER NOT NULL
);
