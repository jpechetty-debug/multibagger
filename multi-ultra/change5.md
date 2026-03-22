# Strategy exposure caps
SWING_MAX_PORTFOLIO_PCT     = 0.30   # 30% max in swing trades
POSITIONAL_MAX_PORTFOLIO_PCT = 0.40  # 40% max in positional
MULTIBAGGER_MAX_PORTFOLIO_PCT = 0.30 # 30% max in multibaggers

# Swing VIX halt (lower than positional)
SWING_VIX_HALT = 25.0

# Multibagger quality filter
MB_MIN_ROE_5Y          = 0.18
MB_MIN_SALES_CAGR_5Y   = 0.18
MB_MAX_DEBT_EQUITY     = 0.50
MB_MIN_CFO_TO_PAT      = 0.75
MB_MAX_MARKET_CAP      = 5_000_00_00_000  # ₹5000 Cr
MB_MIN_PIOTROSKI       = 6
MB_MIN_PROMOTER_PCT    = 45.0
MB_MAX_PLEDGE_PCT      = 5.0

# Swing technical thresholds
SWING_RSI_OVERSOLD     = 35
SWING_RSI_OVERBOUGHT   = 70
SWING_ATR_STOP_MULT    = 1.5
SWING_REWARD_RISK      = 2.0    # 1:2 minimum R/R

# Scheduler cadences
SWING_SCAN_INTERVAL_MIN  = 30   # every 30 min during market hours
POSITIONAL_SCAN_HOUR     = 6    # 6:15 AM daily
MB_SCAN_DAY              = "sun" # weekly Sunday{\rtf1}