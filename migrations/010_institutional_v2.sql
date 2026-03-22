-- @db ops
CREATE TABLE IF NOT EXISTS backtest_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    total_return_pct REAL NOT NULL,
    cagr_pct REAL NOT NULL,
    sharpe REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    win_rate_pct REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    total_costs REAL NOT NULL,
    ending_capital REAL NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_date TEXT NOT NULL,
    audit_json TEXT NOT NULL,
    timestamp INTEGER NOT NULL
);
