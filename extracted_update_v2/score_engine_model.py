"""Composite score engine with heuristic and XGBoost meta scoring.

ModelGuard integration
----------------------
If the ML model has decayed (AUC drop > 5% or return correlation < 0.20),
ModelGuard activates rule-based fallback automatically.  In fallback mode
score_ticker() skips the XGBoost meta-model and uses pure regime-weighted
factor scores from config.BASE_FACTOR_WEIGHTS.  The fallback clears itself
once a fresh model is promoted via ModelGuard.set_baseline().
"""

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
from engines.ml.model_guard import ModelGuard
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
    """Calculates weighted and meta-model scores for a ticker.

    When ModelGuard.is_fallback_active() is True (ML model has decayed),
    score_ticker() automatically switches to _rule_based_score() which
    uses only regime-weighted factor scores — no XGBoost involved.
    """

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
        self._guard = ModelGuard()   # shared singleton across all ScoreEngine instances

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> ScoreResult:
        """Return the composite score for a ticker.

        Routes through rule-based scoring when ModelGuard is in fallback mode.
        """
        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.dependencies.fetcher.fetch(normalized_ticker)

        fundamentals      = self.dependencies.fundamentals.analyze(normalized_ticker, data)
        earnings_revision = self.dependencies.earnings_revision.analyze(normalized_ticker, data)
        momentum          = self.dependencies.momentum.analyze(normalized_ticker, data)
        ownership         = self.dependencies.ownership.analyze(normalized_ticker, data)
        sector_rank       = self.dependencies.sector_rank.analyze(normalized_ticker, data)
        risk_metrics      = self.dependencies.risk_metrics.analyze(normalized_ticker, data)
        regime_result     = self.dependencies.regime.detect()
        weights           = get_weights(regime_result.regime)

        valuation_score = self._valuation_proxy_score(data)
        factor_scores   = [
            FactorScore(factor="fundamentals",      raw_value=fundamentals.score,      normalized_score=fundamentals.score,      weight=weights["fundamentals"]),
            FactorScore(factor="earnings_revision", raw_value=earnings_revision.score, normalized_score=earnings_revision.score, weight=weights["earnings_revision"]),
            FactorScore(factor="momentum",          raw_value=momentum.score,          normalized_score=momentum.score,          weight=weights["momentum"]),
            FactorScore(factor="valuation",         raw_value=valuation_score,         normalized_score=valuation_score,         weight=weights["valuation"]),
            FactorScore(factor="ownership",         raw_value=ownership.score,         normalized_score=ownership.score,         weight=weights["ownership"]),
            FactorScore(factor="sector_strength",   raw_value=sector_rank.score,       normalized_score=sector_rank.score,       weight=weights["sector_strength"]),
            FactorScore(factor="risk",              raw_value=risk_metrics.score,      normalized_score=risk_metrics.score,      weight=weights["risk"]),
        ]
        weighted_score = sum(item.normalized_score * item.weight for item in factor_scores)

        # ── ModelGuard gate ────────────────────────────────────────────────────
        # When the guard is active the ML meta-model is unreliable.
        # Fall back to pure rule-based weighted score immediately.
        if self._guard.is_fallback_active():
            total_score  = self._rule_based_score(factor_scores, weights)
            meta_score   = weighted_score          # report weighted as "meta" for transparency
            ml_mode      = "FALLBACK"
        else:
            feature_vector = self.dependencies.features.build(
                data=data,
                momentum=momentum,
                sector_rank=sector_rank,
                earnings_revision=earnings_revision,
                risk_metrics=risk_metrics,
                regime_result=regime_result,
            )
            meta_score  = self._meta_model_score(feature_vector.to_numpy())
            total_score = weighted_score * (1.0 - config.META_MODEL_BLEND) + meta_score * config.META_MODEL_BLEND
            ml_mode     = "ML"

        total_score = max(0.0, min(100.0, float(total_score)))
        action      = self._action(total_score)
        reasoning   = [
            f"Regime: {regime_result.regime.value} ({regime_result.reason})",
            f"Scoring mode: {ml_mode}",
            f"Weighted score {weighted_score:.1f} | Meta score {meta_score:.1f} | Total {total_score:.1f}",
            f"Momentum {momentum.score:.1f}, Fundamentals {fundamentals.score:.1f}, Risk {risk_metrics.score:.1f}",
        ]
        if ml_mode == "FALLBACK":
            reasoning.insert(1, f"[FALLBACK] Reason: {self._guard.status().get('fallback_reason', 'unknown')}")

        result = ScoreResult(
            ticker=normalized_ticker,
            regime=regime_result.regime,
            weighted_score=weighted_score,
            meta_model_score=meta_score,
            total_score=total_score,
            action=action,
            factor_scores=factor_scores,
            feature_vector=self.dependencies.features.build(
                data=data, momentum=momentum, sector_rank=sector_rank,
                earnings_revision=earnings_revision, risk_metrics=risk_metrics,
                regime_result=regime_result,
            ) if ml_mode == "ML" else factor_scores[0],   # lightweight ref in fallback
            reasoning=reasoning,
            generated_at=int(__import__("time").time()),
        )
        db.save_score_result(result)
        db.log_engine_event("INFO", "engines.score_engine.model", "score computed",
                            {**result.model_dump(), "ml_mode": ml_mode})
        return result

    def _rule_based_score(
        self,
        factor_scores: list[FactorScore],
        weights: dict[str, float],
    ) -> float:
        """Pure rule-based score — no ML.

        Uses BASE_FACTOR_WEIGHTS from config to avoid any regime-weight
        over-concentration during the fallback period.
        """
        base = config.BASE_FACTOR_WEIGHTS
        total_w = sum(base.values()) or 1.0
        score = 0.0
        for fs in factor_scores:
            w = base.get(fs.factor, weights.get(fs.factor, 0.0))
            score += fs.normalized_score * (w / total_w)
        return max(0.0, min(100.0, score))

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
