"""Composite score engine — ModelGuard wired, classifier proba fixed, sector-aware valuation.

Fixes applied
-------------
FIX-1  ModelGuard singleton is now imported and checked at the top of
       score_ticker().  When fallback is active the XGBoost meta-model is
       bypassed entirely and _rule_based_score() is used instead.
       ML mode ("ML" | "FALLBACK") is surfaced in reasoning and the DB log.

FIX-3  _meta_model_score() now calls predict_proba()[:, 1] × 100 on the
       trained SovereignEnsemble classifier instead of predict(), which
       previously returned a binary 0/1 label and caused a bimodal score
       distribution.  The bootstrap fallback is retained for cold-start but
       is an XGBRegressor so predict() is still valid there.

FIX-6  _valuation_proxy_score() now reads the sector-aware PE ceiling from
       config.SECTOR_VALUATION_TEMPLATES so IT stocks are not penalised for
       a PE of 30-35 the same way a value stock at PE 60 would be.

PLACEMENT: replace engines/score_engine/model.py with this file.
"""

from __future__ import annotations

import joblib
import logging
from dataclasses import dataclass
from pathlib import Path
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
from engines.ml.model_guard import ModelGuard          # FIX-1
from engines.score_engine.features import FeatureBuilder
from engines.score_engine.regime import RegimeDetector
from engines.score_engine.weights import get_weights
from models.schemas import FactorScore, FundamentalData, ScoreResult

logger = logging.getLogger(__name__)


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

    Routing
    -------
    When ModelGuard.is_fallback_active() is True (ML model has decayed or
    no trained model exists yet), score_ticker() skips the XGBoost
    meta-model and uses _rule_based_score() — a pure weighted sum from
    config.BASE_FACTOR_WEIGHTS.  This mode is logged as "FALLBACK".

    When a real trained SovereignEnsemble is available (loaded from the
    model registry artifact path) predict_proba()[:,1] × 100 is used.
    This mode is logged as "ML".

    The cold-start bootstrap (random XGBRegressor) is only active until
    the first successful ml-ops --retrain run.
    """

    # Class-level cache — shared across all ScoreEngine instances in a process
    _ensemble_model: Any = None       # fitted SovereignEnsemble (classifier)
    _bootstrap_model: "xgb.XGBRegressor | None" = None  # cold-start fallback

    def __init__(self, dependencies: ScoreDependencies | None = None) -> None:
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
        self._guard = ModelGuard()   # FIX-1: shared singleton

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> ScoreResult:
        """Return the composite score for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = (
            data
            or db.get_fundamental(normalized_ticker, effective=True)
            or self.dependencies.fetcher.fetch(normalized_ticker)
        )

        fundamentals      = self.dependencies.fundamentals.analyze(normalized_ticker, data)
        earnings_revision = self.dependencies.earnings_revision.analyze(normalized_ticker, data)
        momentum          = self.dependencies.momentum.analyze(normalized_ticker, data)
        ownership         = self.dependencies.ownership.analyze(normalized_ticker, data)
        sector_rank       = self.dependencies.sector_rank.analyze(normalized_ticker, data)
        risk_metrics      = self.dependencies.risk_metrics.analyze(normalized_ticker, data)
        regime_result     = self.dependencies.regime.detect()
        weights           = get_weights(regime_result.regime)

        # FIX-6: sector-aware valuation proxy
        valuation_score = self._valuation_proxy_score(data)

        factor_scores = [
            FactorScore(factor="fundamentals",      raw_value=fundamentals.score,      normalized_score=fundamentals.score,      weight=weights["fundamentals"]),
            FactorScore(factor="earnings_revision", raw_value=earnings_revision.score, normalized_score=earnings_revision.score, weight=weights["earnings_revision"]),
            FactorScore(factor="momentum",          raw_value=momentum.score,          normalized_score=momentum.score,          weight=weights["momentum"]),
            FactorScore(factor="valuation",         raw_value=valuation_score,         normalized_score=valuation_score,         weight=weights["valuation"]),
            FactorScore(factor="ownership",         raw_value=ownership.score,         normalized_score=ownership.score,         weight=weights["ownership"]),
            FactorScore(factor="sector_strength",   raw_value=sector_rank.score,       normalized_score=sector_rank.score,       weight=weights["sector_strength"]),
            FactorScore(factor="risk",              raw_value=risk_metrics.score,      normalized_score=risk_metrics.score,      weight=weights["risk"]),
        ]
        weighted_score = sum(fs.normalized_score * fs.weight for fs in factor_scores)

        # ── FIX-1: ModelGuard gate ─────────────────────────────────────────
        if self._guard.is_fallback_active():
            total_score = self._rule_based_score(factor_scores, weights)
            meta_score  = weighted_score   # report weighted as "meta" for transparency
            ml_mode     = "FALLBACK"
        else:
            feature_vector = self.dependencies.features.build(
                data=data,
                momentum=momentum,
                sector_rank=sector_rank,
                earnings_revision=earnings_revision,
                risk_metrics=risk_metrics,
                regime_result=regime_result,
            )
            # FIX-3: use proba-based scoring, not binary predict()
            meta_score  = self._meta_model_score(feature_vector.to_numpy())
            total_score = (
                weighted_score * (1.0 - config.META_MODEL_BLEND)
                + meta_score   * config.META_MODEL_BLEND
            )
            ml_mode = "ML"

        total_score = max(0.0, min(100.0, float(total_score)))
        action      = self._action(total_score)

        reasoning = [
            f"Regime: {regime_result.regime.value} ({regime_result.reason})",
            f"Scoring mode: {ml_mode}",
            f"Weighted {weighted_score:.1f} | Meta {meta_score:.1f} | Total {total_score:.1f}",
            f"Momentum {momentum.score:.1f}, Fundamentals {fundamentals.score:.1f}, Risk {risk_metrics.score:.1f}",
        ]
        if ml_mode == "FALLBACK":
            reasoning.insert(1, f"[FALLBACK] {self._guard.status().get('fallback_reason', 'unknown')}")

        feature_vector_for_result = self.dependencies.features.build(
            data=data, momentum=momentum, sector_rank=sector_rank,
            earnings_revision=earnings_revision, risk_metrics=risk_metrics,
            regime_result=regime_result,
        )

        result = ScoreResult(
            ticker=normalized_ticker,
            regime=regime_result.regime,
            weighted_score=weighted_score,
            meta_model_score=meta_score,
            total_score=total_score,
            action=action,
            factor_scores=factor_scores,
            feature_vector=feature_vector_for_result,
            reasoning=reasoning,
            generated_at=int(__import__("time").time()),
        )
        db.save_score_result(result)
        db.log_engine_event(
            "INFO", "engines.score_engine.model", "score computed",
            {**result.model_dump(), "ml_mode": ml_mode},
        )
        return result

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _rule_based_score(
        self,
        factor_scores: list[FactorScore],
        weights: dict[str, float],
    ) -> float:
        """Pure rule-based score — no ML.

        Uses BASE_FACTOR_WEIGHTS from config to avoid regime-weight
        over-concentration during the fallback period.
        """
        base    = config.BASE_FACTOR_WEIGHTS
        total_w = sum(base.values()) or 1.0
        score   = 0.0
        for fs in factor_scores:
            w      = base.get(fs.factor, weights.get(fs.factor, 0.0))
            score += fs.normalized_score * (w / total_w)
        return max(0.0, min(100.0, score))

    def _valuation_proxy_score(self, data: FundamentalData) -> float:
        """Return a sector-aware valuation proxy.

        FIX-6: PE ceiling is now read from SECTOR_VALUATION_TEMPLATES so
        that high-quality growth sectors (IT: ceiling 32, FMCG: ceiling 34)
        are not penalised for a PE that is normal for their peer group.
        The default ceiling is still 24 for unlisted / unknown sectors.
        """
        template   = config.valuation_template_for_sector(data.sector)
        pe_floor   = float(template.get("pe_floor",   8.0))
        pe_ceiling = float(template.get("pe_ceiling", 24.0))

        # PEG: high PEG = expensive, low PEG = cheap; pivot at 0.5
        peg_score = (
            50.0
            if data.peg_ratio is None
            else max(0.0, min(100.0, 100.0 - (data.peg_ratio - 0.5) * 20.0))
        )

        # PE: score 100 at pe_floor, score 0 at pe_ceiling, linear between
        if data.pe_ratio is None:
            pe_score = 50.0
        else:
            pe_range = max(pe_ceiling - pe_floor, 1.0)
            pe_score = max(0.0, min(100.0, 100.0 - max(0.0, data.pe_ratio - pe_floor) / pe_range * 100.0))

        # Growth bonus: 5-year sales CAGR mapped 0→50 at 0%, 100 at 25%+
        growth_bonus = max(0.0, min(100.0, (data.sales_growth_5y or 0.0) * 200.0 + 50.0))

        return peg_score * 0.4 + pe_score * 0.3 + growth_bonus * 0.3

    # ------------------------------------------------------------------
    # ML model loading — FIX-3: always use predict_proba, not predict
    # ------------------------------------------------------------------

    def _meta_model_score(self, features: np.ndarray) -> float:
        """Return the meta-model score as a probability × 100.

        FIX-3: The trained SovereignEnsemble is a *classifier*.  Calling
        predict() returns binary 0/1 labels that, when blended at 25%,
        corrupt the score distribution.  We call predict_proba()[:,1]
        instead, which gives a calibrated [0, 1] confidence that maps
        cleanly to [0, 100].

        Load order:
        1. Try to load the active trained ensemble from the model registry.
        2. Fall back to a cold-start XGBRegressor (predict() is fine there
           because it was trained as a regressor on synthetic 0–100 labels).
        """
        ensemble = self._load_trained_ensemble()
        if ensemble is not None:
            try:
                X = features.reshape(1, -1)
                # FIX-3: use predict_proba, scale to 0-100
                prob = float(ensemble.predict_proba(X)[0, 1])
                return max(0.0, min(100.0, prob * 100.0))
            except Exception as exc:
                logger.warning("Ensemble predict_proba failed (%s); using bootstrap", exc)

        # Cold-start bootstrap (XGBRegressor — predict() returns continuous score)
        bootstrap = self._get_or_build_bootstrap(features.shape[0])
        prediction = float(bootstrap.predict(features.reshape(1, -1))[0])
        return max(0.0, min(100.0, prediction))

    @classmethod
    def _load_trained_ensemble(cls) -> Any:
        """Load the promoted ensemble model from the model registry artifact.

        Returns None if no trained model has been promoted yet (cold start).
        Caches the loaded object so disk is only hit once per process.
        """
        if cls._ensemble_model is not None:
            return cls._ensemble_model

        try:
            # Ask the ops DB for the active model artifact path
            active = db.get_active_model_version("ensemble_forward_3m_ret")
            if not active:
                return None
            artifact_path = getattr(active, "artifact_path", None) or (
                active.get("artifact_path") if isinstance(active, dict) else None
            )
            if not artifact_path or not Path(artifact_path).exists():
                return None

            model = joblib.load(artifact_path)
            # Validate it has predict_proba (i.e., it's the SovereignEnsemble)
            if not hasattr(model, "predict_proba"):
                logger.warning("Loaded model has no predict_proba — ignoring")
                return None

            cls._ensemble_model = model
            logger.info("ScoreEngine: loaded trained ensemble from %s", artifact_path)
            return cls._ensemble_model
        except Exception as exc:
            logger.debug("Could not load trained ensemble: %s", exc)
            return None

    @classmethod
    def _get_or_build_bootstrap(cls, feature_count: int) -> "xgb.XGBRegressor":
        """Return the deterministic cold-start XGBRegressor bootstrap model."""
        if cls._bootstrap_model is not None:
            return cls._bootstrap_model

        rng     = np.random.default_rng(config.META_MODEL_RANDOM_SEED)
        samples = rng.normal(0.0, 1.0, size=(512, feature_count))
        w       = np.linspace(0.5, 1.5, feature_count)
        labels  = 50 + 12 * np.tanh((samples * w).sum(axis=1) / feature_count) + rng.normal(0, 2, 512)
        labels  = np.clip(labels, 0, 100)

        model = xgb.XGBRegressor(
            n_estimators=40, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=config.META_MODEL_RANDOM_SEED,
        )
        model.fit(samples, labels)
        cls._bootstrap_model = model
        logger.info("ScoreEngine: using cold-start bootstrap model (no trained ensemble found)")
        return cls._bootstrap_model

    @classmethod
    def invalidate_model_cache(cls) -> None:
        """Call after a new model is promoted so the next score_ticker() reloads it."""
        cls._ensemble_model = None

    def _action(self, total_score: float) -> str:
        if total_score >= config.ACTION_THRESHOLDS["BUY"]:
            return "BUY"
        if total_score >= config.ACTION_THRESHOLDS["WATCH"]:
            return "WATCH"
        if total_score >= config.ACTION_THRESHOLDS["WEAK"]:
            return "WEAK"
        return "REJECT"


if __name__ == "__main__":
    print(ScoreEngine().score_ticker("RELIANCE").model_dump())
