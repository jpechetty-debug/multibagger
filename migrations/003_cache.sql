-- @db cache
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT NOT NULL,
    source TEXT NOT NULL,
    ticker TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    PRIMARY KEY (cache_key, source)
);
CREATE INDEX IF NOT EXISTS idx_cache_ticker_source ON cache_entries(ticker, source);
CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at);
