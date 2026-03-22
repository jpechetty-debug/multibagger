-- @db stocks
CREATE TABLE IF NOT EXISTS scores (
    ticker TEXT PRIMARY KEY,
    generated_at INTEGER NOT NULL,
    regime TEXT NOT NULL,
    weighted_score REAL NOT NULL,
    meta_model_score REAL NOT NULL,
    total_score REAL NOT NULL,
    action TEXT NOT NULL,
    factor_scores_json TEXT NOT NULL,
    feature_names_json TEXT NOT NULL,
    feature_values_json TEXT NOT NULL,
    reasoning_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_action ON scores(action, total_score DESC);
CREATE INDEX IF NOT EXISTS idx_scores_generated_at ON scores(generated_at DESC);
