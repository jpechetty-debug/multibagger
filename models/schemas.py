"""Shared Pydantic models and enums."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrategyTag(str, Enum):
    """Trading strategy identifier."""

    SWING = "SWING"
    POSITIONAL = "POSITIONAL"
    MULTIBAGGER = "MULTIBAGGER"


class SwingAction(str, Enum):
    """Swing trade signal actions."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    HOLD = "HOLD"


class AuditableField(str, Enum):
    """Enumerates fields that may be audited or manually corrected."""

    ROE_5Y = "roe_5y"
    ROE_TTM = "roe_ttm"
    SALES_GROWTH_5Y = "sales_growth_5y"
    EPS_GROWTH_TTM = "eps_growth_ttm"
    CFO_TO_PAT = "cfo_to_pat"
    DEBT_EQUITY = "debt_equity"
    PEG_RATIO = "peg_ratio"
    PE_RATIO = "pe_ratio"
    PIOTROSKI_SCORE = "piotroski_score"
    PROMOTER_PCT = "promoter_pct"
    PLEDGE_PCT = "pledge_pct"
    FII_DELTA = "fii_delta"
    DII_DELTA = "dii_delta"
    PRICE = "price"
    MARKET_CAP = "market_cap"
    AVG_VOLUME = "avg_volume"
    SECTOR = "sector"
    UPDATED_AT = "updated_at"

    @classmethod
    def parse(cls, value: str) -> "AuditableField":
        """Parse an enum value from a name or lowercase field key."""

        normalized = value.strip().lower()
        for member in cls:
            if normalized in {member.name.lower(), member.value.lower()}:
                return member
        raise ValueError(f"Unsupported auditable field: {value}")

    @property
    def is_integer(self) -> bool:
        """Return whether the field stores integers."""

        return self is AuditableField.PIOTROSKI_SCORE

    @property
    def is_string(self) -> bool:
        """Return whether the field stores strings."""

        return self is AuditableField.SECTOR

    @property
    def is_numeric(self) -> bool:
        """Return whether the field stores numeric values."""

        return not self.is_string


class FundamentalData(BaseModel):
    """Canonical stored fundamental record."""

    model_config = ConfigDict(use_enum_values=True)

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    price: float | None = None
    market_cap: float | None = None
    avg_volume: float | None = None
    roe_5y: float | None = None
    roe_ttm: float | None = None
    sales_growth_5y: float | None = None
    eps_growth_ttm: float | None = None
    cfo_to_pat: float | None = None
    debt_equity: float | None = None
    peg_ratio: float | None = None
    pe_ratio: float | None = None
    piotroski_score: int | None = None
    promoter_pct: float | None = None
    pledge_pct: float | None = None
    fii_delta: float | None = None
    dii_delta: float | None = None
    updated_at: int = Field(default_factory=lambda: int(time.time()))
    ingestion_quality_score: float = 0.0
    ingestion_issues: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: dict[str, int] = Field(default_factory=dict)
    raw_payload_hash: str | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        """Normalize tickers to uppercase."""

        return value.strip().upper()

    @field_validator("peg_ratio")
    @classmethod
    def _validate_peg_ratio(cls, value: float | None) -> float | None:
        """Reject negative PEG values at the model boundary."""

        if value is not None and value < 0:
            raise ValueError("peg_ratio cannot be negative")
        return value

    @field_validator("piotroski_score")
    @classmethod
    def _validate_piotroski_score(cls, value: int | None) -> int | None:
        """Restrict Piotroski score to the 0-9 range."""

        if value is not None and not 0 <= value <= 9:
            raise ValueError("piotroski_score must be between 0 and 9")
        return value


class SourceSnapshot(BaseModel):
    """Raw or normalized data returned by a single provider."""

    source: str
    ticker: str
    fetched_at: int
    fields: dict[str, Any] = Field(default_factory=dict)


class SourceSnapshotBundle(BaseModel):
    """Collection of provider snapshots for a ticker."""

    ticker: str
    snapshots: list[SourceSnapshot] = Field(default_factory=list)

    def by_source(self) -> dict[str, SourceSnapshot]:
        """Return snapshots indexed by source."""

        return {snapshot.source: snapshot for snapshot in self.snapshots}


class FieldAudit(BaseModel):
    """Single field-level audit result."""

    field_name: AuditableField
    stored_value: Any
    resolved_live_value: Any
    source_name: str
    status: Literal["PASS", "WARN", "FAIL", "MISSING"]
    reason: str
    numeric_delta: float | None = None


class AuditReport(BaseModel):
    """Ticker-level audit report."""

    ticker: str
    run_id: str
    timestamp: int
    overall_status: Literal["PASS", "WARN", "FAIL", "INCOMPLETE"]
    audit_quality_score: float
    fail_count: int
    warn_count: int
    missing_count: int
    field_results: list[FieldAudit] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    triggered_by: str = "manual"

    def to_dataframe(self) -> pd.DataFrame:
        """Return the field results as a DataFrame."""

        return pd.DataFrame(
            [
                {
                    "field_name": result.field_name.value,
                    "stored_value": result.stored_value,
                    "resolved_live_value": result.resolved_live_value,
                    "source_name": result.source_name,
                    "status": result.status,
                    "reason": result.reason,
                    "numeric_delta": result.numeric_delta,
                }
                for result in self.field_results
            ]
        )


class UniverseAuditSummary(BaseModel):
    """Aggregated audit summary across many tickers."""

    tickers_audited: int
    pass_count: int
    warn_count: int
    fail_count: int
    incomplete_count: int
    field_fail_counts: dict[str, int] = Field(default_factory=dict)
    field_warn_counts: dict[str, int] = Field(default_factory=dict)
    field_missing_counts: dict[str, int] = Field(default_factory=dict)
    source_health_alerts: list[str] = Field(default_factory=list)
    average_score: float | None = None
    median_score: float | None = None
    score_distribution: dict[str, int] = Field(default_factory=dict)
    report_rows: list[dict[str, Any]] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return ticker-level summary rows as a DataFrame."""

        return pd.DataFrame(self.report_rows)


class FieldDistribution(BaseModel):
    """Distribution statistics for a single auditable field."""

    field: AuditableField
    min: float
    max: float
    mean: float
    median: float
    p1: float
    p5: float
    p95: float
    p99: float
    outlier_tickers: list[str] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return scalar stats as a single-row DataFrame."""

        return pd.DataFrame(
            [
                {
                    "field": self.field.value,
                    "min": self.min,
                    "max": self.max,
                    "mean": self.mean,
                    "median": self.median,
                    "p1": self.p1,
                    "p5": self.p5,
                    "p95": self.p95,
                    "p99": self.p99,
                }
            ]
        )


class GateResult(BaseModel):
    """Pre-scan gate outcome."""

    passed: bool
    effective_quality_score: float
    skip_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SourceComparisonRow(BaseModel):
    """Comparison view for a single field across providers."""

    field_name: AuditableField
    stored_value: Any
    yfinance_value: Any = None
    nsepython_value: Any = None
    alpha_vantage_value: Any = None
    bse_value: Any = None
    status: Literal["PASS", "WARN", "FAIL", "MISSING"]
    recommended_source: str
    details: str


class SourceComparisonReport(BaseModel):
    """Structured source comparison result for a ticker."""

    ticker: str
    generated_at: int
    rows: list[SourceComparisonRow] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return comparison rows as a DataFrame."""

        return pd.DataFrame(
            [
                {
                    "field_name": row.field_name.value,
                    "stored_value": row.stored_value,
                    "yfinance_value": row.yfinance_value,
                    "nsepython_value": row.nsepython_value,
                    "alpha_vantage_value": row.alpha_vantage_value,
                    "bse_value": row.bse_value,
                    "status": row.status,
                    "recommended_source": row.recommended_source,
                    "details": row.details,
                }
                for row in self.rows
            ]
        )


class MomentumAnalysis(BaseModel):
    """Momentum characteristics for a ticker."""

    ticker: str
    price_return_3m: float
    benchmark_return_3m: float
    relative_strength_3m: float
    price_vs_50dma_pct: float
    volume_acceleration: float
    above_50dma: bool
    score: float
    as_of: int


class FundamentalAnalysis(BaseModel):
    """Fundamental quality analysis."""

    ticker: str
    roe_5y: float | None
    eps_growth_ttm: float | None
    cfo_to_pat: float | None
    piotroski_score: int | None
    piotroski_checks: dict[str, bool] = Field(default_factory=dict)
    score: float
    as_of: int


class OwnershipAnalysis(BaseModel):
    """Ownership quality analysis."""

    ticker: str
    promoter_pct: float | None
    pledge_pct: float | None
    fii_delta: float | None
    dii_delta: float | None
    ownership_clean: bool
    score: float
    as_of: int


class SectorRankAnalysis(BaseModel):
    """Sector-relative ranking and DuPont analysis."""

    ticker: str
    sector: str | None
    peer_count: int
    sector_rank: int
    rank_percentile: float
    top_3: bool
    dupont: dict[str, float | None] = Field(default_factory=dict)
    common_size: dict[str, float | None] = Field(default_factory=dict)
    score: float
    as_of: int


class LiquidityAnalysis(BaseModel):
    """Liquidity profile for a ticker."""

    ticker: str
    avg_daily_volume_20d: float | None
    turnover_value_20d: float | None
    delivery_pct: float | None
    liquidity_ok: bool
    score: float
    as_of: int


class EarningsRevisionAnalysis(BaseModel):
    """Earnings beat and revision analysis."""

    ticker: str
    beat_streak: int
    revision_signal: Literal["UPGRADE", "DOWNGRADE", "NEUTRAL"]
    estimate_trend_pct: float | None
    surprise_mean: float | None
    score: float
    as_of: int


class RiskMetricsAnalysis(BaseModel):
    """Risk metrics computed from price history."""

    ticker: str
    volatility_6m: float | None
    beta_vs_nifty: float | None
    max_drawdown_6m: float | None
    score: float
    as_of: int


class MarketRegime(str, Enum):
    """Supported market regimes."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    QUALITY = "QUALITY"


class RegimeResult(BaseModel):
    """Market regime detection result."""

    regime: MarketRegime
    india_vix: float | None
    breadth_ratio: float | None
    advance_count: int
    decline_count: int
    reason: str
    as_of: int


class ScoreFeatureVector(BaseModel):
    """Feature vector used by the score engine."""

    ticker: str
    feature_names: list[str]
    values: list[float]

    def to_numpy(self) -> np.ndarray:
        """Return the feature vector as a NumPy array."""

        return np.asarray(self.values, dtype=float)

    def to_dict(self) -> dict[str, float]:
        """Return the feature vector as a mapping."""

        return {name: value for name, value in zip(self.feature_names, self.values)}


class FactorScore(BaseModel):
    """Single factor score used in the composite score."""

    factor: str
    raw_value: float
    normalized_score: float
    weight: float


class ScoreResult(BaseModel):
    """Composite score result for a ticker."""

    ticker: str
    regime: MarketRegime
    weighted_score: float
    meta_model_score: float
    total_score: float
    action: Literal["BUY", "WATCH", "WEAK", "REJECT"]
    factor_scores: list[FactorScore] = Field(default_factory=list)
    feature_vector: ScoreFeatureVector
    reasoning: list[str] = Field(default_factory=list)
    generated_at: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return factor-level scores as a DataFrame."""

        return pd.DataFrame(
            [
                {
                    "factor": factor.factor,
                    "raw_value": factor.raw_value,
                    "normalized_score": factor.normalized_score,
                    "weight": factor.weight,
                }
                for factor in self.factor_scores
            ]
        )


class ValuationResult(BaseModel):
    """Intrinsic value estimates for a ticker."""

    ticker: str
    dcf_value: float | None
    eps_value: float | None
    graham_value: float | None
    peg_value: float | None
    fair_value: float | None
    margin_of_safety_pct: float | None
    undervalued: bool
    valuation_confidence: float | None = None
    generated_at: int


class VixState(str, Enum):
    """Portfolio exposure mode based on India VIX."""

    NORMAL = "NORMAL"
    HALF = "HALF"
    HALT = "HALT"


class VixFilterResult(BaseModel):
    """India VIX filter output."""

    vix_value: float | None
    state: VixState
    position_multiplier: float
    reason: str
    as_of: int


class PositionSizingResult(BaseModel):
    """Risk-based position sizing output."""

    ticker: str
    entry_price: float
    stop_loss_price: float
    capital: float
    risk_fraction: float
    kelly_fraction: float
    target_position_value: float
    quantity: int
    conviction: bool
    vix_state: VixState
    as_of: int


class CorrelationFilterResult(BaseModel):
    """Correlation matrix and rejection output."""

    tickers: list[str]
    correlation_matrix: dict[str, dict[str, float]]
    rejected_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    allowed_tickers: list[str] = Field(default_factory=list)
    as_of: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return the correlation matrix as a DataFrame."""

        return pd.DataFrame(self.correlation_matrix)


class PortfolioLimitResult(BaseModel):
    """Portfolio constraints evaluation."""

    passed: bool
    stock_weight_ok: bool
    sector_weight_ok: bool
    liquidity_ok: bool
    correlation_ok: bool
    violations: list[str] = Field(default_factory=list)


class SignalResult(BaseModel):
    """Trading signal output."""

    ticker: str
    action: Literal["BUY", "WATCH", "WEAK", "REJECT"]
    confidence_score: float
    reason_code: str
    strategy_tag: StrategyTag = StrategyTag.POSITIONAL
    satisfied_conditions: list[str] = Field(default_factory=list)
    failed_conditions: list[str] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)
    generated_at: int


class PortfolioPosition(BaseModel):
    """Persisted portfolio position."""

    ticker: str
    sector: str | None
    quantity: int
    avg_cost: float
    last_price: float
    market_value: float
    stop_loss: float
    conviction: bool
    strategy_tag: StrategyTag = StrategyTag.POSITIONAL
    opened_at: int
    updated_at: int


class PortfolioSnapshot(BaseModel):
    """Current portfolio summary."""

    cash: float
    equity_value: float
    total_value: float
    positions: list[PortfolioPosition] = Field(default_factory=list)
    sector_exposure: dict[str, float] = Field(default_factory=dict)
    as_of: int


class BacktestResult(BaseModel):
    """Portfolio backtest result."""

    start: str
    end: str
    win_rate: float
    total_pnl: float
    best_ticker: str | None
    worst_ticker: str | None
    equity_curve: list[dict[str, float | str]] = Field(default_factory=list)
    trade_outcomes: list[dict[str, float | str]] = Field(default_factory=list)


class ModelVersion(BaseModel):
    """Registered ML model metadata."""

    version: str
    model_name: str
    stage: str
    created_at: int
    artifact_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """Forward return prediction and SHAP explanation."""

    ticker: str
    model_version: str
    predicted_forward_return: float
    shap_values: dict[str, float] = Field(default_factory=dict)
    feature_vector: dict[str, float] = Field(default_factory=dict)
    generated_at: int


class PipelineTickerResult(BaseModel):
    """Pipeline output for a single ticker."""

    ticker: str
    action: Literal["BUY", "WATCH", "WEAK", "REJECT", "SKIP", "ERROR", "ENTRY", "EXIT", "HOLD"]
    score: float | None = None
    fair_value: float | None = None
    upside_pct: float | None = None
    strategy_tag: StrategyTag = StrategyTag.POSITIONAL
    data_warnings: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    error: str | None = None
    generated_at: int


class PipelineRunResult(BaseModel):
    """Summary of a pipeline run across many tickers."""

    tickers_requested: int
    processed_count: int
    skipped_count: int
    error_count: int
    action_counts: dict[str, int] = Field(default_factory=dict)
    regime: str | None = None
    results: list[PipelineTickerResult] = Field(default_factory=list)
    started_at: int
    finished_at: int
    summary: str


class TechnicalAnalysis(BaseModel):
    """Technical indicator snapshot for swing trades."""

    ticker: str
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    ema_short: float | None = None
    ema_long: float | None = None
    atr: float | None = None
    trend_bullish: bool = False
    as_of: int


class BreakoutResult(BaseModel):
    """Breakout scanner output."""

    ticker: str
    volume_surge: bool = False
    near_52w_high: bool = False
    consolidation_breakout: bool = False
    volume_ratio: float | None = None
    pct_from_52w_high: float | None = None
    breakout_score: float = 0.0
    as_of: int


class StopTarget(BaseModel):
    """ATR-based stop-loss and target price."""

    entry_price: float
    stop_loss: float
    target_price: float
    atr: float
    reward_risk: float


class SwingSignal(BaseModel):
    """Complete swing trade signal."""

    ticker: str
    action: SwingAction
    strategy_tag: StrategyTag = StrategyTag.SWING
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    atr: float | None = None
    rsi: float | None = None
    macd_signal_value: float | None = None
    breakout_score: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    generated_at: int


class QualityResult(BaseModel):
    """Multibagger quality filter output."""

    ticker: str
    passed: bool
    roe_ok: bool = False
    cagr_ok: bool = False
    debt_ok: bool = False
    cfo_ok: bool = False
    mcap_ok: bool = False
    piotroski_ok: bool = False
    promoter_ok: bool = False
    pledge_ok: bool = False
    quality_score: float = 0.0
    fail_reasons: list[str] = Field(default_factory=list)
    as_of: int


class EarlySignalResult(BaseModel):
    """Early signal detection for multibagger candidates."""

    ticker: str
    promoter_buying: bool = False
    fii_entry: bool = False
    earnings_beats: int = 0
    early_signal_score: float = 0.0
    signals: list[str] = Field(default_factory=list)
    as_of: int


class TAMResult(BaseModel):
    """TAM and tailwind scoring for multibagger candidates."""

    ticker: str
    sector_tailwind: bool = False
    tam_runway_score: float = 0.0
    sector: str | None = None
    revenue_to_mcap_ratio: float | None = None
    as_of: int


class MultibaggerCandidate(BaseModel):
    """Final multibagger candidate with conviction and tranche plan."""

    ticker: str
    strategy_tag: StrategyTag = StrategyTag.MULTIBAGGER
    quality_score: float = 0.0
    early_signal_score: float = 0.0
    tam_score: float = 0.0
    conviction_score: float = 0.0
    tranche_plan: list[dict[str, float]] = Field(default_factory=list)
    action: Literal["BUY", "WATCH", "REJECT"] = "REJECT"
    reasoning: list[str] = Field(default_factory=list)
    generated_at: int


class StrategySizingResult(BaseModel):
    """Strategy-aware position sizing output."""

    ticker: str
    strategy_tag: StrategyTag
    risk_pct: float
    position_value: float
    quantity: int
    entry_price: float
    stop_loss: float
    as_of: int


class ExposureLimitResult(BaseModel):
    """Per-strategy exposure limit check."""

    strategy_tag: StrategyTag
    current_exposure_pct: float
    proposed_exposure_pct: float
    max_allowed_pct: float
    within_limit: bool
    headroom_pct: float


def parse_field_value(field: AuditableField, raw_value: str) -> Any:
    """Parse CLI or UI input for a specific auditable field."""

    if field.is_string:
        return raw_value.strip()
    if field.is_integer:
        return int(float(raw_value))
    return float(raw_value)


if __name__ == "__main__":
    sample = FundamentalData(ticker="reliance", ingestion_quality_score=88.0)
    print(sample.model_dump())
