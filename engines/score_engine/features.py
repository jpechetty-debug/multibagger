"""Feature vector builder for the score engine."""

from __future__ import annotations

from typing import Any

import config
from models.schemas import (
    EarningsRevisionAnalysis,
    FundamentalData,
    MomentumAnalysis,
    RegimeResult,
    RiskMetricsAnalysis,
    ScoreFeatureVector,
    SectorRankAnalysis,
)


class FeatureBuilder:
    """Builds model features from fundamental and analysis inputs."""

    def build(
        self,
        data: FundamentalData,
        momentum: MomentumAnalysis,
        sector_rank: SectorRankAnalysis,
        earnings_revision: EarningsRevisionAnalysis,
        risk_metrics: RiskMetricsAnalysis,
        regime_result: RegimeResult,
    ) -> ScoreFeatureVector:
        """Return a deterministic feature vector."""

        feature_map: dict[str, float] = {
            "roe_5y": float(data.roe_5y or 0.0),
            "roe_ttm": float(data.roe_ttm or 0.0),
            "sales_growth_5y": float(data.sales_growth_5y or 0.0),
            "eps_growth_ttm": float(data.eps_growth_ttm or 0.0),
            "cfo_to_pat": float(data.cfo_to_pat or 0.0),
            "debt_equity": float(data.debt_equity or 0.0),
            "peg_ratio": float(data.peg_ratio or 0.0),
            "pe_ratio": float(data.pe_ratio or 0.0),
            "piotroski_score": float(data.piotroski_score or 0.0),
            "promoter_pct": float(data.promoter_pct or 0.0),
            "pledge_pct": float(data.pledge_pct or 0.0),
            "fii_delta": float(data.fii_delta or 0.0),
            "dii_delta": float(data.dii_delta or 0.0),
            "price_return_3m": momentum.price_return_3m,
            "relative_strength_3m": momentum.relative_strength_3m,
            "price_vs_50dma_pct": momentum.price_vs_50dma_pct,
            "volume_acceleration": momentum.volume_acceleration,
            "sector_rank_pct": sector_rank.rank_percentile,
            "beat_streak": float(earnings_revision.beat_streak),
            "estimate_trend_pct": float(earnings_revision.estimate_trend_pct or 0.0),
            "volatility_6m": float(risk_metrics.volatility_6m or 0.0),
            "beta_vs_nifty": float(risk_metrics.beta_vs_nifty or 0.0),
            "max_drawdown_6m": float(risk_metrics.max_drawdown_6m or 0.0),
            "india_vix": float(regime_result.india_vix or 0.0),
            "breadth_ratio": float(regime_result.breadth_ratio or 0.0),
        }
        return ScoreFeatureVector(
            ticker=data.ticker,
            feature_names=list(config.FEATURE_NAMES),
            values=[feature_map[name] for name in config.FEATURE_NAMES],
        )


if __name__ == "__main__":
    print("FeatureBuilder import ok")
