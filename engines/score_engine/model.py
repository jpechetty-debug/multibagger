"""Composite score engine with heuristic and XGBoost meta scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.analysis.earnings_revision import EarningsRevisionAnalyzer
from engines.analysis.fundamentals import FundamentalAnalyzer
from engines.analysis.momentum import MomentumAnalyzer
from engines.analysis.ownership import OwnershipAnalyzer
from engines.analysis.risk_metrics import RiskMetricsAnalyzer
from engines.analysis.sector_rank import SectorRankAnalyzer
from engines.score_engine.features import FeatureBuilder
from engines.score_engine.regime import RegimeDetector
from engines.score_engine.weights import get_weights
from models.schemas import FactorScore, FundamentalData, ScoreResult


@dataclass
class ScoreDependencies:
    """Dependency bundle for the score engine."""

    fetcher: DataFetcher
    fundamentals: FundamentalAnalyzer
    earnings_revision: EarningsRevisionAnalyzer
    momentum: MomentumAnalyzer
    ownership: OwnershipAnalyzer
    sector_rank: SectorRankAnalyzer
    risk_metrics: RiskMetricsAnalyzer
    regime: RegimeDetector
    features: FeatureBuilder


class ScoreEngine:
    """Calculates weighted and meta-model scores for a ticker."""

    _meta_model: xgb.XGBRegressor | None = None

    def __init__(self, dependencies: ScoreDependencies | None = None) -> None:
        """Initialize the score engine."""

        if dependencies is None:
            fetcher = DataFetcher()
            dependencies = ScoreDependencies(
                fetcher=fetcher,
                fundamentals=FundamentalAnalyzer(fetcher),
                earnings_revision=EarningsRevisionAnalyzer(fetcher),
                momentum=MomentumAnalyzer(fetcher),
                ownership=OwnershipAnalyzer(fetcher),
                sector_rank=SectorRankAnalyzer(fetcher),
                risk_metrics=RiskMetricsAnalyzer(fetcher),
                regime=RegimeDetector(),
                features=FeatureBuilder(),
            )
        self.dependencies = dependencies

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> ScoreResult:
        """Return the composite score for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.dependencies.fetcher.fetch(normalized_ticker)
        fundamentals = self.dependencies.fundamentals.analyze(normalized_ticker, data)
        earnings_revision = self.dependencies.earnings_revision.analyze(normalized_ticker, data)
        momentum = self.dependencies.momentum.analyze(normalized_ticker, data)
        ownership = self.dependencies.ownership.analyze(normalized_ticker, data)
        sector_rank = self.dependencies.sector_rank.analyze(normalized_ticker, data)
        risk_metrics = self.dependencies.risk_metrics.analyze(normalized_ticker, data)
        regime_result = self.dependencies.regime.detect()
        weights = get_weights(regime_result.regime)

        valuation_score = self._valuation_proxy_score(data)
        factor_scores = [
            FactorScore(factor="fundamentals", raw_value=fundamentals.score, normalized_score=fundamentals.score, weight=weights["fundamentals"]),
            FactorScore(factor="earnings_revision", raw_value=earnings_revision.score, normalized_score=earnings_revision.score, weight=weights["earnings_revision"]),
            FactorScore(factor="momentum", raw_value=momentum.score, normalized_score=momentum.score, weight=weights["momentum"]),
            FactorScore(factor="valuation", raw_value=valuation_score, normalized_score=valuation_score, weight=weights["valuation"]),
            FactorScore(factor="ownership", raw_value=ownership.score, normalized_score=ownership.score, weight=weights["ownership"]),
            FactorScore(factor="sector_strength", raw_value=sector_rank.score, normalized_score=sector_rank.score, weight=weights["sector_strength"]),
            FactorScore(factor="risk", raw_value=risk_metrics.score, normalized_score=risk_metrics.score, weight=weights["risk"]),
        ]
        weighted_score = sum(item.normalized_score * item.weight for item in factor_scores)

        feature_vector = self.dependencies.features.build(
            data=data,
            momentum=momentum,
            sector_rank=sector_rank,
            earnings_revision=earnings_revision,
            risk_metrics=risk_metrics,
            regime_result=regime_result,
        )
        meta_model_score = self._meta_model_score(feature_vector.to_numpy())
        total_score = weighted_score * (1.0 - config.META_MODEL_BLEND) + meta_model_score * config.META_MODEL_BLEND
        total_score = max(0.0, min(100.0, float(total_score)))
        action = self._action(total_score)
        reasoning = [
            f"Regime: {regime_result.regime.value} ({regime_result.reason})",
            f"Weighted score {weighted_score:.1f} blended with meta-model score {meta_model_score:.1f}",
            f"Momentum {momentum.score:.1f}, Fundamentals {fundamentals.score:.1f}, Risk {risk_metrics.score:.1f}",
        ]

        result = ScoreResult(
            ticker=normalized_ticker,
            regime=regime_result.regime,
            weighted_score=weighted_score,
            meta_model_score=meta_model_score,
            total_score=total_score,
            action=action,
            factor_scores=factor_scores,
            feature_vector=feature_vector,
            reasoning=reasoning,
            generated_at=int(__import__("time").time()),
        )
        db.save_score_result(result)
        db.log_engine_event("INFO", "engines.score_engine.model", "score computed", result.model_dump())
        return result

    def _valuation_proxy_score(self, data: FundamentalData) -> float:
        """Return a valuation proxy until the dedicated valuation engine exists."""

        peg_score = 50.0 if data.peg_ratio is None else max(0.0, min(100.0, 100.0 - (data.peg_ratio - 0.5) * 20.0))
        pe_score = 50.0 if data.pe_ratio is None else max(0.0, min(100.0, 100.0 - max(0.0, data.pe_ratio - 10.0) * 2.0))
        growth_bonus = max(0.0, min(100.0, (data.sales_growth_5y or 0.0) * 200.0 + 50.0))
        return peg_score * 0.4 + pe_score * 0.3 + growth_bonus * 0.3

    def _meta_model_score(self, features: np.ndarray) -> float:
        """Return the XGBoost meta-model score."""

        if ScoreEngine._meta_model is None:
            ScoreEngine._meta_model = self._bootstrap_model(features.shape[0])
        prediction = float(ScoreEngine._meta_model.predict(features.reshape(1, -1))[0])
        return max(0.0, min(100.0, prediction))

    def _bootstrap_model(self, feature_count: int) -> xgb.XGBRegressor:
        """Train a deterministic bootstrap model for meta scoring."""

        rng = np.random.default_rng(config.META_MODEL_RANDOM_SEED)
        samples = rng.normal(loc=0.0, scale=1.0, size=(512, feature_count))
        weights = np.linspace(0.5, 1.5, feature_count)
        labels = 50 + 12 * np.tanh((samples * weights).sum(axis=1) / feature_count) + rng.normal(0, 2, size=512)
        labels = np.clip(labels, 0, 100)
        model = xgb.XGBRegressor(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=config.META_MODEL_RANDOM_SEED,
        )
        model.fit(samples, labels)
        return model

    def _action(self, total_score: float) -> str:
        """Map total score to a human-readable action."""

        if total_score >= config.ACTION_THRESHOLDS["BUY"]:
            return "BUY"
        if total_score >= config.ACTION_THRESHOLDS["WATCH"]:
            return "WATCH"
        if total_score >= config.ACTION_THRESHOLDS["WEAK"]:
            return "WEAK"
        return "REJECT"


if __name__ == "__main__":
    print(ScoreEngine().score_ticker("RELIANCE").model_dump())
