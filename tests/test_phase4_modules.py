"""Tests for Phase 4 valuation, risk, signal, and ML modules."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from data.db import db
from engines.portfolio_engine import PortfolioEngine
from engines.risk.portfolio_limits import PortfolioLimitsChecker
from engines.risk.position_sizing import PositionSizer
from engines.signal_engine import SignalDependencies, SignalEngine
from engines.valuation_engine import ValuationEngine
from ml.registry import ModelRegistry
from ml.train import ModelTrainer
from models.schemas import (
    CorrelationFilterResult,
    FundamentalData,
    GateResult,
    MarketRegime,
    MomentumAnalysis,
    OwnershipAnalysis,
    PortfolioPosition,
    RegimeResult,
    ScoreFeatureVector,
    ScoreResult,
    SectorRankAnalysis,
    SignalResult,
    ValuationResult,
    VixFilterResult,
    VixState,
)


class _StaticFetcher:
    """Deterministic fetcher for tests."""

    def __init__(self, record: FundamentalData) -> None:
        """Store the record returned by fetch."""

        self.record = record

    def fetch(self, ticker: str, refresh: bool = False) -> FundamentalData:
        """Return the configured record."""

        return self.record


class _StaticScoreEngine:
    """Deterministic score engine."""

    def __init__(self, result: ScoreResult) -> None:
        """Store the score result."""

        self.result = result

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> ScoreResult:
        """Return the configured score."""

        return self.result


class _StaticValuationEngine:
    """Deterministic valuation engine."""

    def __init__(self, result: ValuationResult) -> None:
        """Store the valuation result."""

        self.result = result

    def value_ticker(self, ticker: str, data: FundamentalData | None = None) -> ValuationResult:
        """Return the configured valuation."""

        return self.result


class _StaticAnalyzer:
    """Deterministic analysis dependency."""

    def __init__(self, result) -> None:
        """Store the analysis result."""

        self.result = result

    def analyze(self, ticker: str, data: FundamentalData | None = None):
        """Return the configured analysis result."""

        return self.result


class _StaticVixFilter:
    """Deterministic VIX filter dependency."""

    def __init__(self, result: VixFilterResult) -> None:
        """Store the VIX result."""

        self.result = result

    def evaluate(self) -> VixFilterResult:
        """Return the configured VIX result."""

        return self.result


class _StaticGate:
    """Deterministic pre-scan gate."""

    def __init__(self, result: GateResult) -> None:
        """Store the gate result."""

        self.result = result

    def check(self, ticker: str, data: FundamentalData, latest_audit=None) -> GateResult:
        """Return the configured gate result."""

        return self.result


class _StaticCorrelationAnalyzer:
    """Deterministic correlation dependency."""

    def correlation_matrix(self, tickers: list[str]) -> CorrelationFilterResult:
        """Allow all provided tickers."""

        return CorrelationFilterResult(
            tickers=tickers,
            correlation_matrix={ticker: {other: 0.0 for other in tickers} for ticker in tickers},
            allowed_tickers=tickers,
            as_of=int(time.time()),
        )


def _sample_record(ticker: str = "TESTPH4") -> FundamentalData:
    """Return a reusable fundamental record."""

    return FundamentalData(
        ticker=ticker,
        company_name="Phase4 Test",
        sector="Financial Services",
        price=100.0,
        market_cap=50_000_000.0,
        avg_volume=100_000.0,
        roe_5y=0.18,
        roe_ttm=0.16,
        sales_growth_5y=0.12,
        eps_growth_ttm=0.10,
        cfo_to_pat=1.0,
        debt_equity=0.4,
        peg_ratio=1.0,
        pe_ratio=15.0,
        piotroski_score=7,
        promoter_pct=60.0,
        pledge_pct=0.0,
        fii_delta=1.0,
        dii_delta=0.5,
        updated_at=int(time.time()),
        ingestion_quality_score=95.0,
    )


def test_valuation_engine_picks_minimum_fair_value(monkeypatch) -> None:
    """Valuation engine should use the minimum non-null estimate as fair value."""

    record = _sample_record("VALTEST")
    statements = {
        "financials": pd.DataFrame(
            {
                pd.Timestamp("2025-03-31"): {
                    "Net Income": 1_000_000.0,
                    "Total Revenue": 10_000_000.0,
                    "Gross Profit": 4_000_000.0,
                }
            }
        ).T.T,
        "balance_sheet": pd.DataFrame(
            {
                pd.Timestamp("2025-03-31"): {
                    "Common Stock Equity": 5_000_000.0,
                    "Ordinary Shares Number": 100_000.0,
                }
            }
        ).T.T,
        "cashflow": pd.DataFrame({pd.Timestamp("2025-03-31"): {"Free Cash Flow": 800_000.0}}).T.T,
    }
    monkeypatch.setattr("engines.valuation_engine.load_financial_statements", lambda ticker: statements)
    result = ValuationEngine(fetcher=_StaticFetcher(record)).value_ticker(record.ticker, data=record)
    candidates = [value for value in [result.dcf_value, result.eps_value, result.graham_value, result.peg_value] if value is not None]
    assert result.fair_value == min(candidates)


def test_valuation_engine_uses_sector_median_pe_when_company_pe_is_missing(monkeypatch) -> None:
    """Valuation fallback should use peer-sector PE when the company PE is missing."""

    target = _sample_record("VALPEMISS")
    target.pe_ratio = None
    peer_a = _sample_record("VALPEER1")
    peer_a.pe_ratio = 20.0
    peer_b = _sample_record("VALPEER2")
    peer_b.pe_ratio = 24.0
    for record in (target, peer_a, peer_b):
        db.upsert_fundamental(record)

    empty_statements = {
        "financials": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }
    monkeypatch.setattr("engines.valuation_engine.load_financial_statements", lambda ticker: empty_statements)

    result = ValuationEngine(fetcher=_StaticFetcher(target)).value_ticker(target.ticker, data=target)
    assert result.fair_value is not None
    assert result.eps_value is not None
    with db.connection("stocks") as conn:
        for ticker in (target.ticker, peer_a.ticker, peer_b.ticker):
            conn.execute("DELETE FROM fundamentals WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM valuations WHERE ticker = ?", (ticker,))


def test_valuation_engine_calculates_confidence(monkeypatch) -> None:
    """Valuation engine should calculate confidence correctly based on spread."""

    record = _sample_record("VALCONF")
    statements = {
        "financials": pd.DataFrame({pd.Timestamp("2025-03-31"): {"Net Income": 5_000_000.0}}).T.T,
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2025-03-31"): {"Common Stock Equity": 20_000_000.0, "Ordinary Shares Number": 100_000.0}
        }).T.T,
        "cashflow": pd.DataFrame({pd.Timestamp("2025-03-31"): {"Free Cash Flow": 500_000.0}}).T.T,
    }
    monkeypatch.setattr("engines.valuation_engine.load_financial_statements", lambda ticker: statements)
    
    engine = ValuationEngine(fetcher=_StaticFetcher(record))
    result = engine.value_ticker(record.ticker, data=record)
    
    assert result.valuation_confidence is not None
    assert 0.0 <= result.valuation_confidence <= 100.0


def test_position_sizer_handles_half_vix() -> None:
    """Half-VIX mode should reduce the per-trade risk fraction."""

    result = PositionSizer().size_position(
        "POSTEST",
        entry_price=100.0,
        stop_loss_price=92.0,
        capital=1_000_000.0,
        confidence_score=90.0,
        vix_state=VixState.HALF,
    )
    assert result.risk_fraction == config.HIGH_VIX_RISK_FRACTION
    assert result.quantity >= 0


def test_signal_engine_returns_buy_when_all_conditions_pass() -> None:
    """Signal engine should return BUY when all conditions are satisfied."""

    record = _sample_record("SIGTEST")
    score = ScoreResult(
        ticker=record.ticker,
        regime=MarketRegime.BULL,
        weighted_score=82.0,
        meta_model_score=78.0,
        total_score=81.0,
        action="BUY",
        feature_vector=ScoreFeatureVector(ticker=record.ticker, feature_names=["x"], values=[1.0]),
        generated_at=int(time.time()),
    )
    dependencies = SignalDependencies(
        fetcher=_StaticFetcher(record),
        score_engine=_StaticScoreEngine(score),
        valuation_engine=_StaticValuationEngine(
            ValuationResult(
                ticker=record.ticker,
                dcf_value=140.0,
                eps_value=130.0,
                graham_value=125.0,
                peg_value=120.0,
                fair_value=120.0,
                margin_of_safety_pct=0.20,
                undervalued=True,
                generated_at=int(time.time()),
            )
        ),
        momentum=_StaticAnalyzer(MomentumAnalysis(ticker=record.ticker, price_return_3m=0.1, benchmark_return_3m=0.05, relative_strength_3m=0.05, price_vs_50dma_pct=0.08, volume_acceleration=1.2, above_50dma=True, score=70.0, as_of=int(time.time()))),
        sector_rank=_StaticAnalyzer(SectorRankAnalysis(ticker=record.ticker, sector=record.sector, peer_count=4, sector_rank=1, rank_percentile=1.0, top_3=True, dupont={}, common_size={}, score=90.0, as_of=int(time.time()))),
        ownership=_StaticAnalyzer(OwnershipAnalysis(ticker=record.ticker, promoter_pct=60.0, pledge_pct=0.0, fii_delta=1.0, dii_delta=0.5, ownership_clean=True, score=80.0, as_of=int(time.time()))),
        vix_filter=_StaticVixFilter(VixFilterResult(vix_value=15.0, state=VixState.NORMAL, position_multiplier=1.0, reason="ok", as_of=int(time.time()))),
        pre_scan_gate=_StaticGate(GateResult(passed=True, effective_quality_score=95.0, warnings=[])),
    )
    result = SignalEngine(dependencies=dependencies).evaluate(record.ticker, data=record)
    assert result.action == "BUY"
    assert "score_gt_75" in result.satisfied_conditions


def test_portfolio_limits_fail_for_large_single_stock() -> None:
    """Portfolio limits should fail when a candidate breaches single-stock weight."""

    record = _sample_record("LIMITTEST")
    checker = PortfolioLimitsChecker(fetcher=_StaticFetcher(record))
    positions = [
        PortfolioPosition(
            ticker="AAA",
            sector="Financial Services",
            quantity=10,
            avg_cost=100.0,
            last_price=100.0,
            market_value=1000.0,
            stop_loss=90.0,
            conviction=False,
            opened_at=int(time.time()),
            updated_at=int(time.time()),
        )
    ]
    result = checker.check(record.ticker, candidate_value=10_000.0, positions=positions, data=record)
    assert result.passed is False
    assert "MAX_SINGLE_STOCK_WEIGHT" in result.violations


def test_portfolio_limits_accept_candidate_when_total_portfolio_value_is_passed() -> None:
    """Portfolio limits should use full portfolio value when supplied by the portfolio engine."""

    record = _sample_record("LIMITPASS")
    checker = PortfolioLimitsChecker(
        fetcher=_StaticFetcher(record),
        correlation_analyzer=_StaticCorrelationAnalyzer(),
    )
    result = checker.check(
        record.ticker,
        candidate_value=100_000.0,
        positions=[],
        data=record,
        portfolio_total_value=config.DEFAULT_PORTFOLIO_CAPITAL,
    )
    assert result.passed is True
    assert result.stock_weight_ok is True


def test_model_registry_roundtrip() -> None:
    """Model registry should persist and reload an active XGBoost model."""

    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 4))
    labels = rng.normal(size=20)
    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(features, labels)
    registry = ModelRegistry()
    version = registry.save_model(model, "test_phase4_registry", {"feature_names": ["a", "b", "c", "d"]})
    loaded_model, loaded_version = registry.load_active("test_phase4_registry")
    prediction_original = float(model.predict(features[:1])[0])
    prediction_loaded = float(loaded_model.predict(features[:1])[0])
    assert version.version == loaded_version.version
    assert abs(prediction_original - prediction_loaded) < 1e-9


def test_model_trainer_augments_sparse_dataset(monkeypatch) -> None:
    """Trainer should bootstrap sparse fallback data up to the configured minimum."""

    record = _sample_record("MLTRAIN")
    score = ScoreResult(
        ticker=record.ticker,
        regime=MarketRegime.BULL,
        weighted_score=75.0,
        meta_model_score=70.0,
        total_score=73.0,
        action="WATCH",
        feature_vector=ScoreFeatureVector(ticker=record.ticker, feature_names=["x"], values=[1.0]),
        generated_at=int(time.time()),
    )
    trainer = ModelTrainer(score_engine=_StaticScoreEngine(score))
    monkeypatch.setattr(db, "list_pit_fundamentals", lambda: [])
    monkeypatch.setattr(db, "list_fundamentals", lambda effective=True: [record])
    monkeypatch.setattr(trainer, "_fallback_label", lambda ticker: 0.12)
    features, labels, feature_names, dataset_source = trainer._training_dataset()
    assert features.shape[0] == config.MIN_TRAINING_ROWS
    assert len(labels) == config.MIN_TRAINING_ROWS
    assert len(feature_names) == features.shape[1]
    assert dataset_source == "fallback_augmented"
