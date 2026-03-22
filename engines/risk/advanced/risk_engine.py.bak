"""
engines/risk/advanced/risk_engine.py
--------------------------------------
Three interlocking risk modules:

1. CorrelationController
   Detects and flags correlated position pairs before entry.

2. VolatilityScaler
   Adjusts position size inversely to recent realised volatility.

3. FactorNeutralizer
   Measures each signal's factor exposures and penalises over-concentration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1. Correlation Controller
# ---------------------------------------------------------------------------

@dataclass
class CorrelationFlag:
    ticker_a: str
    ticker_b: str
    correlation: float
    severity: str       # "HIGH" | "MODERATE"
    recommendation: str


class CorrelationController:
    """
    Computes pairwise correlations across a candidate set and flags
    pairs that would create concentrated factor exposures.
    """

    HIGH_THRESHOLD     = 0.75
    MODERATE_THRESHOLD = 0.55

    def analyse(
        self,
        tickers: list[str],
        returns_df: pd.DataFrame,
        lookback: int = 60,
    ) -> list[CorrelationFlag]:
        available = [t for t in tickers if t in returns_df.columns]
        if len(available) < 2:
            return []

        ret = returns_df[available].tail(lookback).dropna()
        if len(ret) < 20:
            return []

        corr = ret.corr()
        flags: list[CorrelationFlag] = []

        for i, t_a in enumerate(available):
            for t_b in available[i+1:]:
                if t_a not in corr.index or t_b not in corr.columns:
                    continue
                rho = float(corr.loc[t_a, t_b])
                if rho >= self.HIGH_THRESHOLD:
                    flags.append(CorrelationFlag(
                        ticker_a=t_a, ticker_b=t_b,
                        correlation=round(rho, 3),
                        severity="HIGH",
                        recommendation=f"Drop {t_b} — {rho:.2f} correlation with {t_a}.",
                    ))
                elif rho >= self.MODERATE_THRESHOLD:
                    flags.append(CorrelationFlag(
                        ticker_a=t_a, ticker_b=t_b,
                        correlation=round(rho, 3),
                        severity="MODERATE",
                        recommendation=f"Size both down — {rho:.2f} correlation.",
                    ))

        return sorted(flags, key=lambda f: f.correlation, reverse=True)

    def filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        returns_df: pd.DataFrame,
        lookback: int = 60,
        max_correlation: float = 0.75,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        sorted_cands = sorted(
            candidates, key=lambda c: c.get("total_score", 0), reverse=True
        )
        tickers = [c["ticker"] for c in sorted_cands]
        available = [t for t in tickers if t in returns_df.columns]

        if len(available) < 2:
            return candidates, []

        ret = returns_df[available].tail(lookback).dropna()
        if len(ret) < 20:
            return candidates, []

        corr = ret.corr()
        kept: list[str] = []
        dropped: list[str] = []

        for ticker in tickers:
            if ticker not in available:
                kept.append(ticker)
                continue
            too_correlated = any(
                ticker in corr.index
                and k in corr.columns
                and corr.loc[ticker, k] >= max_correlation
                for k in kept if k in corr.columns
            )
            if too_correlated:
                dropped.append(ticker)
            else:
                kept.append(ticker)

        kept_set = set(kept)
        return [c for c in candidates if c["ticker"] in kept_set], dropped


# ---------------------------------------------------------------------------
# 2. Volatility Scaler
# ---------------------------------------------------------------------------

@dataclass
class VolatilityScaledPosition:
    ticker: str
    base_weight: float
    scaled_weight: float
    realised_vol: float
    vol_scalar: float
    rupee_risk_estimate: float


class VolatilityScaler:
    """Scales position sizes inversely to realised volatility."""

    TRADING_DAYS = 252

    def __init__(self, target_daily_vol_pct: float = 1.5, lookback: int = 20):
        self.target_daily_vol = target_daily_vol_pct / 100.0
        self.lookback = lookback

    def scale(
        self,
        tickers: list[str],
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> list[VolatilityScaledPosition]:
        results: list[VolatilityScaledPosition] = []
        raw_scaled: dict[str, float] = {}

        for ticker in tickers:
            base_w = weights.get(ticker, 0)
            if base_w <= 0:
                continue

            if ticker in returns_df.columns:
                recent = returns_df[ticker].tail(self.lookback).dropna()
                if len(recent) >= 5:
                    daily_vol = float(recent.std())
                    ann_vol   = daily_vol * np.sqrt(self.TRADING_DAYS)
                else:
                    daily_vol = self.target_daily_vol
                    ann_vol   = daily_vol * np.sqrt(self.TRADING_DAYS)
            else:
                daily_vol = self.target_daily_vol
                ann_vol   = daily_vol * np.sqrt(self.TRADING_DAYS)

            scalar    = _clamp_vol(self.target_daily_vol / (daily_vol + 1e-9), 0.3, 1.5)
            scaled_w  = base_w * scalar
            raw_scaled[ticker] = scaled_w

            rupee_risk = 100_000 * daily_vol

            results.append(VolatilityScaledPosition(
                ticker=ticker,
                base_weight=round(base_w, 4),
                scaled_weight=0.0,
                realised_vol=round(ann_vol * 100, 2),
                vol_scalar=round(scalar, 3),
                rupee_risk_estimate=round(rupee_risk, 0),
            ))

        total = sum(raw_scaled.values())
        for pos in results:
            pos.scaled_weight = round(raw_scaled[pos.ticker] / total, 4) if total > 0 else 0.0

        return sorted(results, key=lambda p: p.scaled_weight, reverse=True)

    def portfolio_vol(
        self,
        tickers: list[str],
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> float:
        available = [t for t in tickers if t in returns_df.columns and weights.get(t, 0) > 0]
        if len(available) < 2:
            return 0.0
        ret = returns_df[available].tail(self.lookback).dropna()
        if len(ret) < 10:
            return 0.0
        w   = np.array([weights.get(t, 0) for t in available])
        w   = w / w.sum()
        cov = ret.cov().values * self.TRADING_DAYS
        return float(np.sqrt(w @ cov @ w))


# ---------------------------------------------------------------------------
# 3. Factor Neutralizer
# ---------------------------------------------------------------------------

FACTOR_COLUMNS = {
    "value":    ["pe_ratio", "pb_ratio", "margin_of_safety_pct"],
    "momentum": ["price_return_3m", "relative_strength_3m", "price_vs_50dma_pct"],
    "quality":  ["roic_current", "roe", "total_score"],
    "size":     ["market_cap"],
}

MAX_FACTOR_TILT = 1.5


@dataclass
class FactorExposure:
    factor: str
    portfolio_zscore: float
    benchmark_zscore: float
    tilt: float
    breached: bool
    message: str


@dataclass
class FactorReport:
    exposures: list[FactorExposure]
    any_breached: bool
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "any_breached": self.any_breached,
            "exposures": [
                {
                    "factor":  e.factor,
                    "tilt":    round(e.tilt, 3),
                    "breached": e.breached,
                    "message": e.message,
                }
                for e in self.exposures
            ],
            "recommendations": self.recommendations,
        }


class FactorNeutralizer:
    """Measures factor exposures and flags excessive tilts."""

    def __init__(self, max_tilt: float = MAX_FACTOR_TILT):
        self.max_tilt = max_tilt

    def analyse(
        self,
        candidates: list[dict[str, Any]],
        weights: dict[str, float],
        universe_df: pd.DataFrame,
    ) -> FactorReport:
        exposures: list[FactorExposure] = []
        recommendations: list[str] = []

        for factor, columns in FACTOR_COLUMNS.items():
            available_cols = [c for c in columns if c in universe_df.columns]
            if not available_cols:
                continue

            universe_factor = universe_df[available_cols].mean(axis=1)
            bm_mean = universe_factor.mean()
            bm_std  = universe_factor.std()
            if bm_std < 1e-9:
                continue

            portfolio_z = 0.0
            total_weight = 0.0
            for cand in candidates:
                ticker = cand.get("ticker", "")
                w      = weights.get(ticker, 0)
                if w <= 0:
                    continue
                factor_vals = [cand.get(c) for c in available_cols if cand.get(c) is not None]
                if not factor_vals:
                    continue
                raw_score = np.mean(factor_vals)
                z_score   = (raw_score - bm_mean) / bm_std

                if factor in ("value", "size"):
                    z_score = -z_score

                portfolio_z  += w * z_score
                total_weight += w

            if total_weight > 0:
                portfolio_z /= total_weight

            tilt     = portfolio_z
            breached = abs(tilt) > self.max_tilt

            direction = "overweight" if tilt > 0 else "underweight"
            msg = (
                f"Portfolio is {direction} {factor} by {abs(tilt):.2f} z-scores."
            )

            exposures.append(FactorExposure(
                factor=factor,
                portfolio_zscore=round(portfolio_z, 3),
                benchmark_zscore=0.0,
                tilt=round(tilt, 3),
                breached=breached,
                message=msg,
            ))

            if breached:
                recommendations.append(f"Reduce {factor} tilt.")

        return FactorReport(
            exposures=exposures,
            any_breached=any(e.breached for e in exposures),
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Composite risk check
# ---------------------------------------------------------------------------

@dataclass
class RiskCheckResult:
    correlation_flags:  list[CorrelationFlag]
    volatility_scaled:  list[VolatilityScaledPosition]
    factor_report:      FactorReport
    portfolio_vol:      float
    risk_summary:       str
    action_required:    bool

    def to_dict(self) -> dict:
        return {
            "portfolio_vol_annualised_pct": round(self.portfolio_vol * 100, 2),
            "correlation_high_flags":  sum(1 for f in self.correlation_flags if f.severity == "HIGH"),
            "correlation_moderate_flags": sum(1 for f in self.correlation_flags if f.severity == "MODERATE"),
            "factor_breaches":  self.factor_report.any_breached,
            "action_required":  self.action_required,
            "risk_summary":     self.risk_summary,
            "factor_report":    self.factor_report.to_dict(),
        }


class AdvancedRiskEngine:
    """Single entry point for correlation, vol-scaling, and factor checks."""

    def __init__(
        self,
        max_correlation: float = 0.75,
        target_daily_vol_pct: float = 1.5,
        max_factor_tilt: float = MAX_FACTOR_TILT,
    ):
        self.cc = CorrelationController()
        self.vs = VolatilityScaler(target_daily_vol_pct=target_daily_vol_pct)
        self.fn = FactorNeutralizer(max_tilt=max_factor_tilt)
        self.max_corr = max_correlation

    def check(
        self,
        candidates: list[dict[str, Any]],
        weights: dict[str, float],
        returns_df: pd.DataFrame,
        universe_df: pd.DataFrame,
    ) -> RiskCheckResult:
        tickers = [c["ticker"] for c in candidates]
        corr_flags = self.cc.analyse(tickers, returns_df)
        high_flags = [f for f in corr_flags if f.severity == "HIGH"]
        vol_scaled = self.vs.scale(tickers, weights, returns_df)
        port_vol   = self.vs.portfolio_vol(tickers, weights, returns_df)
        factor_report = self.fn.analyse(candidates, weights, universe_df)

        issues: list[str] = []
        if high_flags:
            issues.append(f"{len(high_flags)} high-correlation pair(s)")
        if port_vol > 0.25:
            issues.append(f"Vol {port_vol*100:.1f}% > 25%")
        if factor_report.any_breached:
            issues.append("Factor tilt breach")

        action_required = bool(issues)
        summary = "RISK ALERT: " + " | ".join(issues) if issues else f"Vol {port_vol*100:.1f}%"

        return RiskCheckResult(
            correlation_flags=corr_flags,
            volatility_scaled=vol_scaled,
            factor_report=factor_report,
            portfolio_vol=port_vol,
            risk_summary=summary,
            action_required=action_required,
        )

def _clamp_vol(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
