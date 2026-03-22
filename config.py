"""Application configuration for the Sovereign audit-first slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT_DIR / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
EXPORT_DIR = RUNTIME_DIR / "exports"

DB_PATH = RUNTIME_DIR / "stocks.db"
SWING_DB_PATH = RUNTIME_DIR / "swing_trades.db"
MB_DB_PATH = RUNTIME_DIR / "multibaggers.db"
PIT_PATH = RUNTIME_DIR / "pit_store.db"
CACHE_PATH = RUNTIME_DIR / "cache.db"
OPS_PATH = RUNTIME_DIR / "ops.db"

BACKUPS_DIR = RUNTIME_DIR / "backups"
BACKUP_RETENTION_DAYS = 7

# ---------------------------------------------------------------------------
# Strategy exposure caps
# ---------------------------------------------------------------------------
SWING_MAX_PORTFOLIO_PCT = 0.30
POSITIONAL_MAX_PORTFOLIO_PCT = 0.40
MULTIBAGGER_MAX_PORTFOLIO_PCT = 0.30

# ---------------------------------------------------------------------------
# Swing trade configuration
# ---------------------------------------------------------------------------
SWING_VIX_HALT = 25.0
SWING_RSI_OVERSOLD = 35
SWING_RSI_OVERBOUGHT = 70
SWING_ATR_STOP_MULT = 1.5
SWING_REWARD_RISK = 2.0
SWING_RSI_PERIOD = 14
SWING_MACD_FAST = 12
SWING_MACD_SLOW = 26
SWING_MACD_SIGNAL = 9
SWING_BB_PERIOD = 20
SWING_BB_STD = 2
SWING_EMA_SHORT = 9
SWING_EMA_LONG = 21
SWING_VOLUME_SURGE_MULT = 2.0
SWING_52W_HIGH_PROXIMITY_PCT = 0.05
SWING_CONSOLIDATION_DAYS = 20
SWING_CONSOLIDATION_RANGE_PCT = 0.08

# ---------------------------------------------------------------------------
# Multibagger quality filter thresholds
# ---------------------------------------------------------------------------
MB_MIN_ROE_5Y = 0.18
MB_MIN_SALES_CAGR_5Y = 0.18
MB_MAX_DEBT_EQUITY = 0.50
MB_MIN_CFO_TO_PAT = 0.75
MB_MAX_MARKET_CAP = 5_000_00_00_000  # ₹5,000 Cr
MB_MIN_PIOTROSKI = 6
MB_MIN_PROMOTER_PCT = 45.0
MB_MAX_PLEDGE_PCT = 5.0
MB_TRANCHE_COUNT = 3
MB_TRANCHE_RISK_PCT = 0.005

# ---------------------------------------------------------------------------
# Scheduler cadences
# ---------------------------------------------------------------------------
SWING_SCAN_INTERVAL_MIN = 30
POSITIONAL_SCAN_HOUR = 6
MB_SCAN_DAY = "sun"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

ENV_PATH = ROOT_DIR / ".env"

WARN_STALE_DAYS = 2
FAIL_STALE_DAYS = 7

FAIL_PENALTY = 20
WARN_PENALTY = 5
MIN_DATA_QUALITY = 50

CACHE_TTL_PRICE = 300
CACHE_TTL_FUNDAMENTALS = 86400
CACHE_TTL_OWNERSHIP = 86400
CACHE_TTL_VIX = 300
CACHE_TTL_ALPHA_VANTAGE = 86400
CACHE_TTL_HISTORY = 3600
CACHE_TTL_FINANCIALS = 21600

SOURCE_NAME_YFINANCE = "yfinance"
SOURCE_NAME_NSEPYTHON = "nsepython"
SOURCE_NAME_ALPHA_VANTAGE = "alpha_vantage"
SOURCE_NAME_INDIA_VIX = "india_vix"
SOURCE_NAME_BSE = "bse"
SOURCE_NAME_STORED = "stored"
SOURCE_NAME_OVERRIDE = "manual_override"

SOURCE_PRECEDENCE: dict[str, tuple[str, ...]] = {
    "market": (SOURCE_NAME_NSEPYTHON, SOURCE_NAME_YFINANCE, SOURCE_NAME_ALPHA_VANTAGE),
    "ownership": (SOURCE_NAME_BSE, SOURCE_NAME_NSEPYTHON, SOURCE_NAME_ALPHA_VANTAGE),
    "fundamentals": (SOURCE_NAME_ALPHA_VANTAGE, SOURCE_NAME_YFINANCE, SOURCE_NAME_NSEPYTHON),
    "price_drift_check": (SOURCE_NAME_YFINANCE,),
}

NSE_SECTOR_LIST = [
    "Automobile and Auto Components",
    "Capital Goods",
    "Chemicals",
    "Construction",
    "Construction Materials",
    "Consumer Durables",
    "Consumer Services",
    "Diversified",
    "Fast Moving Consumer Goods",
    "Financial Services",
    "Forest Materials",
    "Healthcare",
    "Information Technology",
    "Media Entertainment and Publication",
    "Metals and Mining",
    "Oil Gas and Consumable Fuels",
    "Power",
    "Realty",
    "Services",
    "Telecommunication",
    "Textiles",
]

SUPPORTED_UNIVERSE_PRESETS = ("QUICK", "STANDARD", "EXTENDED", "NIFTY100", "NIFTY200", "SMALLCAP", "SECTORS")
NIFTY_YFINANCE_SYMBOL = "^NSEI"
DEFAULT_SCAN_UNIVERSE = "STANDARD"
PIPELINE_MAX_CONCURRENCY = 5

SCHEDULER_TIMEZONE = "Asia/Kolkata"
DAILY_SCAN_HOUR = 6
DAILY_SCAN_MINUTE = 15
DAILY_REPORT_HOUR = 18
DAILY_REPORT_MINUTE = 0
WEEKLY_RETRAIN_DAY = "sun"
WEEKLY_RETRAIN_HOUR = 2
WEEKLY_RETRAIN_MINUTE = 0
CACHE_EVICT_HOUR = 3
CACHE_EVICT_MINUTE = 0
DB_OPTIMIZE_HOUR = 3
DB_OPTIMIZE_MINUTE = 30

VIX_BULL_MAX = 18.0
VIX_SIDEWAYS_MAX = 25.0
VIX_BEAR_MIN = 30.0
BREADTH_BULL_MIN = 1.20
BREADTH_BEAR_MAX = 0.80
BREADTH_QUALITY_MIN = 0.95
BREADTH_QUALITY_MAX = 1.10

ALERT_SCORE_CHANGE_THRESHOLD = 10.0
ALERT_FAIR_VALUE_TOLERANCE_PCT = 0.02
ALERT_VIX_SPIKE_THRESHOLD = VIX_BEAR_MIN
TELEGRAM_API_TIMEOUT = 10

PRICE_HISTORY_PERIOD = "1y"
PRICE_HISTORY_INTERVAL = "1d"
MOMENTUM_LOOKBACK_DAYS = 63
DMA_WINDOW = 50
VOLUME_SHORT_WINDOW = 10
VOLUME_LONG_WINDOW = 50
RISK_LOOKBACK_DAYS = 126

BASE_FACTOR_WEIGHTS = {
    "fundamentals": 0.25,
    "earnings_revision": 0.15,
    "momentum": 0.15,
    "valuation": 0.15,
    "ownership": 0.10,
    "sector_strength": 0.10,
    "risk": 0.10,
}

REGIME_FACTOR_WEIGHTS = {
    "BULL": {
        "fundamentals": 0.20,
        "earnings_revision": 0.15,
        "momentum": 0.25,
        "valuation": 0.10,
        "ownership": 0.10,
        "sector_strength": 0.10,
        "risk": 0.10,
    },
    "BEAR": {
        "fundamentals": 0.30,
        "earnings_revision": 0.10,
        "momentum": 0.05,
        "valuation": 0.15,
        "ownership": 0.10,
        "sector_strength": 0.05,
        "risk": 0.25,
    },
    "SIDEWAYS": {
        "fundamentals": 0.25,
        "earnings_revision": 0.15,
        "momentum": 0.10,
        "valuation": 0.15,
        "ownership": 0.10,
        "sector_strength": 0.10,
        "risk": 0.15,
    },
    "QUALITY": {
        "fundamentals": 0.30,
        "earnings_revision": 0.10,
        "momentum": 0.10,
        "valuation": 0.15,
        "ownership": 0.15,
        "sector_strength": 0.05,
        "risk": 0.15,
    },
}

FEATURE_NAMES = [
    "roe_5y",
    "roe_ttm",
    "sales_growth_5y",
    "eps_growth_ttm",
    "cfo_to_pat",
    "debt_equity",
    "peg_ratio",
    "pe_ratio",
    "piotroski_score",
    "promoter_pct",
    "pledge_pct",
    "fii_delta",
    "dii_delta",
    "price_return_3m",
    "relative_strength_3m",
    "price_vs_50dma_pct",
    "volume_acceleration",
    "sector_rank_pct",
    "beat_streak",
    "estimate_trend_pct",
    "volatility_6m",
    "beta_vs_nifty",
    "max_drawdown_6m",
    "india_vix",
    "breadth_ratio",
]

META_MODEL_BLEND = 0.25
META_MODEL_RANDOM_SEED = 42

ACTION_THRESHOLDS = {
    "BUY": 80.0,
    "WATCH": 70.0,
    "WEAK": 50.0,
}

DCF_DISCOUNT_RATE = 0.12
DCF_TERMINAL_GROWTH = 0.04
DCF_PROJECTION_YEARS = 5
EPS_VALUATION_DEFAULT_PE = 18.0
GRAHAM_NUMBER_FACTOR = 22.5
PEG_TARGET = 1.0
MARGIN_OF_SAFETY_FACTOR = 0.80
VALUATION_CONFIDENCE_ACTIONABLE_MIN = 55.0
VALUATION_CONFIDENCE_LOW_MAX = 40.0
VALUATION_MODEL_SPREAD_WARN = 1.75
VALUATION_MODEL_SPREAD_QUARANTINE = 3.00

SECTOR_VALUATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "__default__": {
        "label": "Core",
        "discount_rate": DCF_DISCOUNT_RATE,
        "terminal_growth": DCF_TERMINAL_GROWTH,
        "projection_years": DCF_PROJECTION_YEARS,
        "pe_floor": 10.0,
        "pe_ceiling": 24.0,
        "growth_floor": 0.02,
        "growth_ceiling": 0.12,
        "fair_value_floor_ratio": 0.50,
        "fair_value_ceiling_ratio": 3.00,
        "peer_floor_ratio": 0.45,
        "peer_ceiling_ratio": 2.25,
        "fallback_order": ("EPS", "DCF", "Graham", "PEG"),
    },
    "Financial Services": {
        "label": "Financials",
        "discount_rate": 0.13,
        "terminal_growth": 0.04,
        "pe_floor": 8.0,
        "pe_ceiling": 18.0,
        "growth_ceiling": 0.10,
        "fair_value_ceiling_ratio": 2.20,
        "peer_ceiling_ratio": 1.80,
    },
    "Information Technology": {
        "label": "Technology",
        "discount_rate": 0.12,
        "terminal_growth": 0.05,
        "pe_floor": 14.0,
        "pe_ceiling": 32.0,
        "growth_ceiling": 0.18,
        "fair_value_ceiling_ratio": 3.40,
        "peer_ceiling_ratio": 2.40,
    },
    "Fast Moving Consumer Goods": {
        "label": "Consumer Staples",
        "discount_rate": 0.11,
        "terminal_growth": 0.05,
        "pe_floor": 18.0,
        "pe_ceiling": 34.0,
        "growth_ceiling": 0.12,
        "fair_value_ceiling_ratio": 2.80,
        "peer_floor_ratio": 0.55,
    },
    "Healthcare": {
        "label": "Healthcare",
        "discount_rate": 0.12,
        "terminal_growth": 0.05,
        "pe_floor": 14.0,
        "pe_ceiling": 28.0,
        "growth_ceiling": 0.15,
        "fair_value_ceiling_ratio": 3.00,
    },
    "Capital Goods": {
        "label": "Industrial",
        "discount_rate": 0.13,
        "terminal_growth": 0.04,
        "pe_floor": 12.0,
        "pe_ceiling": 24.0,
        "growth_ceiling": 0.14,
        "fair_value_ceiling_ratio": 2.60,
    },
    "Metals and Mining": {
        "label": "Cyclicals",
        "discount_rate": 0.14,
        "terminal_growth": 0.03,
        "pe_floor": 6.0,
        "pe_ceiling": 14.0,
        "growth_floor": 0.00,
        "growth_ceiling": 0.08,
        "fair_value_ceiling_ratio": 2.00,
        "peer_ceiling_ratio": 1.60,
    },
    "Oil Gas and Consumable Fuels": {
        "label": "Energy",
        "discount_rate": 0.14,
        "terminal_growth": 0.03,
        "pe_floor": 7.0,
        "pe_ceiling": 16.0,
        "growth_floor": 0.00,
        "growth_ceiling": 0.08,
        "fair_value_ceiling_ratio": 2.10,
        "peer_ceiling_ratio": 1.70,
    },
}

VIX_HALF_THRESHOLD = 25.0
VIX_HALT_THRESHOLD = 35.0

NORMAL_RISK_FRACTION = 0.01
CONVICTION_RISK_FRACTION = 0.02
HIGH_VIX_RISK_FRACTION = 0.005
MAX_KELLY_FRACTION = 0.25
DEFAULT_REWARD_TO_RISK = 2.0

MAX_SINGLE_STOCK_WEIGHT = 0.15
MAX_SECTOR_WEIGHT = 0.25
CORRELATION_LIMIT = 0.70
MIN_LIQUIDITY_VALUE = 10_000_000.0
DEFAULT_PORTFOLIO_CAPITAL = 1_000_000.0
DEFAULT_STOP_LOSS_PCT = 0.08
DEFAULT_REBALANCE_TOLERANCE = 0.03

MODEL_DIR = RUNTIME_DIR / "models"
DEFAULT_MODEL_NAME = "forward_return_xgb"
ACTIVE_MODEL_STAGE = "active"
FORWARD_RETURN_HORIZON_DAYS = 21
MIN_TRAINING_ROWS = 8
TRAINING_FEATURE_AUGMENT_STD = 0.01
TRAINING_LABEL_AUGMENT_STD = 0.002

FIELD_AUDIT_RULES: dict[str, dict[str, Any]] = {
    "roe_5y": {"min": -0.50, "max": 0.80, "warn_high": 0.60},
    "roe_ttm": {"min": -1.00, "max": 1.00},
    "sales_growth_5y": {"min": -0.30, "max": 1.00, "warn_high": 0.80},
    "eps_growth_ttm": {"min": -2.00, "max": 5.00, "warn_high": 3.00},
    "cfo_to_pat": {"min": 0.00, "max": 5.00, "warn_low": 0.30},
    "debt_equity": {"min": 0.00, "max": 10.00, "warn_high": 3.00, "fail_high": 8.00},
    "peg_ratio": {"min": 0.10, "max": 10.00},
    "pe_ratio": {"min": 0.00, "max": 200.00},
    "piotroski_score": {"min": 0, "max": 9},
    "promoter_pct": {"min": 0.0, "max": 90.0, "warn_low": 20.0},
    "pledge_pct": {"min": 0.0, "max": 100.0, "warn_high": 10.0, "fail_high": 30.0},
    "price": {"min_exclusive": 0.0},
    "market_cap": {"min_exclusive": 10_000_000.0},
    "avg_volume": {"min_exclusive": 10_000.0},
}


def valuation_template_for_sector(sector: str | None) -> dict[str, Any]:
    """Return the sector-aware valuation template with defaults applied."""

    template = dict(SECTOR_VALUATION_TEMPLATES["__default__"])
    if sector:
        template.update(SECTOR_VALUATION_TEMPLATES.get(sector, {}))
    return template

CRITICAL_INGESTION_FIELDS = (
    "price",
    "market_cap",
    "avg_volume",
    "roe_5y",
    "roe_ttm",
    "sales_growth_5y",
    "eps_growth_ttm",
    "promoter_pct",
    "pledge_pct",
    "updated_at",
)

INGESTION_MISSING_PENALTY = 8
INGESTION_STALE_PENALTY = 10
INGESTION_CONFLICT_PENALTY = 15

REQUIRED_IMPORTS = (
    "typer",
    "rich",
    "pydantic",
    "pandas",
    "streamlit",
    "plotly",
    "yfinance",
    "requests",
    "apscheduler",
)


def _load_env_file(env_path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv-style file."""

    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


_ENV_VALUES = _load_env_file(ENV_PATH)


def get_env(key: str, default: str | None = None) -> str | None:
    """Return an environment variable from the process or .env file."""

    return os.getenv(key) or _ENV_VALUES.get(key, default)


ALPHA_VANTAGE_API_KEY = get_env("ALPHA_VANTAGE_API_KEY")
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by the application."""

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def source_ttl(source: str) -> int:
    """Return the configured cache TTL for a source."""

    mapping = {
        SOURCE_NAME_YFINANCE: CACHE_TTL_FUNDAMENTALS,
        SOURCE_NAME_NSEPYTHON: CACHE_TTL_PRICE,
        SOURCE_NAME_ALPHA_VANTAGE: CACHE_TTL_ALPHA_VANTAGE,
        SOURCE_NAME_INDIA_VIX: CACHE_TTL_VIX,
        SOURCE_NAME_BSE: CACHE_TTL_OWNERSHIP,
        "history_1y": CACHE_TTL_HISTORY,
        "benchmark_history_1y": CACHE_TTL_HISTORY,
        "financials_annual": CACHE_TTL_FINANCIALS,
        "earnings_meta": CACHE_TTL_HISTORY,
    }
    return mapping.get(source, CACHE_TTL_FUNDAMENTALS)


if __name__ == "__main__":
    ensure_runtime_dirs()
    print(
        {
            "root": str(ROOT_DIR),
            "runtime": str(RUNTIME_DIR),
            "dbs": [str(DB_PATH), str(PIT_PATH), str(CACHE_PATH), str(OPS_PATH)],
            "alpha_vantage_key_present": bool(ALPHA_VANTAGE_API_KEY),
        }
    )
