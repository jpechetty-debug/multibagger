"""Tests for Phase 3 analysis and score engine contracts."""

from __future__ import annotations

import time

from data.db import db
from engines.score_engine.features import FeatureBuilder
from engines.score_engine.model import ScoreDependencies, ScoreEngine
from engines.score_engine.weights import get_weights
from models.schemas import (
    EarningsRevisionAnalysis,
    FactorScore,
    FundamentalData,
    FundamentalAnalysis,
    MarketRegime,
    MomentumAnalysis,
    OwnershipAnalysis,
    RegimeResult,
    RiskMetricsAnalysis,
    ScoreFeatureVector,
    SectorRankAnalysis,
)


class _StaticAnalyzer:
    """Small test double that always returns the same analysis result."""

    def __init__(self, result) -> None:
        """Store the result returned by analyze."""

        self.result = result

    def analyze(self, ticker: str, data=None):
        """Return the configured result."""

        return self.result


class _StaticFeatureBuilder:
    """Deterministic feature builder for tests."""

    def __init__(self, result: ScoreFeatureVector) -> None:
        """Store the feature vector returned by build."""

        self.result = result

    def build(self, **kwargs) -> ScoreFeatureVector:
        """Return the configured feature vector."""

        return self.result


class _StaticRegimeDetector:
    """Deterministic regime detector for tests."""

    def __init__(self, result: RegimeResult) -> None:
        """Store the result returned by detect."""

        self.result = result

    def detect(self) -> RegimeResult:
        """Return the configured regime result."""

        return self.result


class _StaticFetcher:
    """Fetcher test double."""

    def __init__(self, record: FundamentalData) -> None:
        """Store the record returned by fetch."""

        self.record = record

    def fetch(self, ticker: str, refresh: bool = False) -> FundamentalData:
        """Return the configured record."""

        return self.record


def test_regime_weights_sum_to_one() -> None:
    """Every regime weight profile should remain normalized."""

    for regime in MarketRegime:
        weights = get_weights(regime)
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_feature_builder_outputs_configured_feature_count() -> None:
    """Feature builder should emit the configured feature vector."""

    record = FundamentalData(
        ticker="TESTF",
        company_name="Feature Test",
        sector="Financial Services",
        price=100.0,
        market_cap=20_000_000.0,
        avg_volume=50_000.0,
        roe_5y=0.15,
        roe_ttm=0.14,
        sales_growth_5y=0.11,
        eps_growth_ttm=0.09,
        cfo_to_pat=0.8,
        debt_equity=0.5,
        peg_ratio=1.1,
        pe_ratio=14.0,
        piotroski_score=6,
        promoter_pct=55.0,
        pledge_pct=0.0,
        fii_delta=1.0,
        dii_delta=0.5,
        updated_at=int(time.time()),
        ingestion_quality_score=90.0,
    )
    vector = FeatureBuilder().build(
        data=record,
        momentum=MomentumAnalysis(
            ticker="TESTF",
            price_return_3m=0.12,
            benchmark_return_3m=0.08,
            relative_strength_3m=0.04,
            price_vs_50dma_pct=0.05,
            volume_acceleration=1.2,
            above_50dma=True,
            score=62.0,
            as_of=int(time.time()),
        ),
        sector_rank=SectorRankAnalysis(
            ticker="TESTF",
            sector="Financial Services",
            peer_count=5,
            sector_rank=2,
            rank_percentile=0.8,
            top_3=True,
            dupont={},
            common_size={},
            score=75.0,
            as_of=int(time.time()),
        ),
        earnings_revision=EarningsRevisionAnalysis(
            ticker="TESTF",
            beat_streak=2,
            revision_signal="UPGRADE",
            estimate_trend_pct=0.03,
            surprise_mean=5.0,
            score=68.0,
            as_of=int(time.time()),
        ),
        risk_metrics=RiskMetricsAnalysis(
            ticker="TESTF",
            volatility_6m=0.25,
            beta_vs_nifty=0.9,
            max_drawdown_6m=-0.12,
            score=70.0,
            as_of=int(time.time()),
        ),
        regime_result=RegimeResult(
            regime=MarketRegime.BULL,
            india_vix=14.0,
            breadth_ratio=1.3,
            advance_count=1200,
            decline_count=900,
            reason="test",
            as_of=int(time.time()),
        ),
    )
    assert len(vector.feature_names) == len(vector.values)
    assert vector.ticker == "TESTF"


def test_score_engine_produces_and_persists_score() -> None:
    """Score engine should combine factor scores and persist the latest score row."""

    ticker = "TESTSCORE"
    record = FundamentalData(
        ticker=ticker,
        company_name="Score Test",
        sector="Financial Services",
        price=100.0,
        market_cap=25_000_000.0,
        avg_volume=60_000.0,
        roe_5y=0.16,
        roe_ttm=0.14,
        sales_growth_5y=0.10,
        eps_growth_ttm=0.08,
        cfo_to_pat=0.9,
        debt_equity=0.4,
        peg_ratio=1.0,
        pe_ratio=15.0,
        piotroski_score=7,
        promoter_pct=58.0,
        pledge_pct=0.0,
        fii_delta=1.5,
        dii_delta=0.8,
        updated_at=int(time.time()),
        ingestion_quality_score=95.0,
    )
    db.upsert_fundamental(record)
    dependencies = ScoreDependencies(
        fetcher=_StaticFetcher(record),
        fundamentals=_StaticAnalyzer(FundamentalAnalysis(ticker=ticker, roe_5y=0.16, eps_growth_ttm=0.08, cfo_to_pat=0.9, piotroski_score=7, piotroski_checks={}, score=82.0, as_of=int(time.time()))),
        earnings_revision=_StaticAnalyzer(EarningsRevisionAnalysis(ticker=ticker, beat_streak=2, revision_signal="UPGRADE", estimate_trend_pct=0.04, surprise_mean=7.0, score=70.0, as_of=int(time.time()))),
        momentum=_StaticAnalyzer(MomentumAnalysis(ticker=ticker, price_return_3m=0.12, benchmark_return_3m=0.08, relative_strength_3m=0.04, price_vs_50dma_pct=0.05, volume_acceleration=1.1, above_50dma=True, score=68.0, as_of=int(time.time()))),
        ownership=_StaticAnalyzer(OwnershipAnalysis(ticker=ticker, promoter_pct=58.0, pledge_pct=0.0, fii_delta=1.5, dii_delta=0.8, ownership_clean=True, score=80.0, as_of=int(time.time()))),
        sector_rank=_StaticAnalyzer(SectorRankAnalysis(ticker=ticker, sector="Financial Services", peer_count=4, sector_rank=1, rank_percentile=1.0, top_3=True, dupont={}, common_size={}, score=85.0, as_of=int(time.time()))),
        risk_metrics=_StaticAnalyzer(RiskMetricsAnalysis(ticker=ticker, volatility_6m=0.22, beta_vs_nifty=0.95, max_drawdown_6m=-0.10, score=72.0, as_of=int(time.time()))),
        regime=_StaticRegimeDetector(RegimeResult(regime=MarketRegime.BULL, india_vix=15.0, breadth_ratio=1.4, advance_count=1500, decline_count=900, reason="test", as_of=int(time.time()))),
        features=_StaticFeatureBuilder(ScoreFeatureVector(ticker=ticker, feature_names=["a", "b", "c"], values=[0.1, 0.2, 0.3])),
    )
    engine = ScoreEngine(dependencies=dependencies)
    engine._meta_model_score = lambda _: 64.0  # type: ignore[method-assign]
    result = engine.score_ticker(ticker, data=record)
    assert result.action in {"WATCH", "BUY"}
    latest = db.get_latest_score(ticker)
    assert latest is not None
    assert latest["ticker"] == ticker
    assert abs(latest["total_score"] - result.total_score) < 1e-9
