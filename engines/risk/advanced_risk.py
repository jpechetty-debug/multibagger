"""
Three advanced risk modules bundled in one file for clean imports.
VaREngine         â€” Historical and parametric Value-at-Risk + CVaR
DrawdownGuard     â€” Live drawdown monitoring with exposure reduction rules
LiquidityFilter   â€” Position sizing relative to average daily volume
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# 1. VaR Engine
# ===========================================================================

@dataclass
class VaRResult:
    confidence:      float          # e.g. 0.95
    horizon_days:    int
    historical_var:  float          # as fraction e.g. 0.032 = 3.2% loss
    parametric_var:  float
    cvar:            float          # Conditional VaR (Expected Shortfall)
    portfolio_vol:   float          # annualised
    limit_breached:  bool
    limit:           float
    message:         str

    def to_dict(self) -> dict:
        return {
            "confidence":     self.confidence,
            "horizon_days":   self.horizon_days,
            "historical_var_pct": round(self.historical_var * 100, 2),
            "parametric_var_pct": round(self.parametric_var * 100, 2),
            "cvar_pct":       round(self.cvar * 100, 2),
            "portfolio_vol_ann_pct": round(self.portfolio_vol * 100, 2),
            "limit_breached": self.limit_breached,
            "message":        self.message,
        }


class VaREngine:
    """
    Computes portfolio VaR and CVaR at configurable confidence levels.
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        var_limit:        float = 0.020,   # 2% 1-day 95% VaR limit
        cvar_limit:       float = 0.035,   # 3.5% CVaR limit
        confidence:       float = 0.95,
        horizon_days:     int   = 1,
        lookback:         int   = 252,
        reduction_factor: float = 0.60,    # reduce to 60% of weights on breach
    ):
        self.var_limit        = var_limit
        self.cvar_limit       = cvar_limit
        self.confidence       = confidence
        self.horizon_days     = horizon_days
        self.lookback         = lookback
        self.reduction_factor = reduction_factor

    def compute(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> VaRResult:
        tickers   = [t for t in weights if t in returns_df.columns and weights[t] > 0]
        if not tickers:
            return self._empty_result()

        w   = np.array([weights[t] for t in tickers])
        w   = w / w.sum()
        ret = returns_df[tickers].tail(self.lookback).dropna()

        if len(ret) < 30:
            return self._empty_result("Insufficient return history")

        # â”€â”€ Portfolio return series â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        port_ret = (ret * w).sum(axis=1)

        # â”€â”€ Historical VaR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cutoff      = int(len(port_ret) * (1 - self.confidence))
        sorted_ret  = np.sort(port_ret.values)
        h_var       = float(-sorted_ret[max(cutoff - 1, 0)])
        h_var      *= math.sqrt(self.horizon_days)

        # â”€â”€ CVaR (Expected Shortfall) â€” mean of losses beyond VaR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        tail_losses = -sorted_ret[:max(cutoff, 1)]
        cvar        = float(tail_losses.mean()) * math.sqrt(self.horizon_days)

        # â”€â”€ Parametric VaR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        mu          = port_ret.mean()
        sigma       = port_ret.std()
        z           = abs(float(np.percentile(np.random.standard_normal(10000),
                                               (1 - self.confidence) * 100)))
        p_var       = float((-mu + z * sigma) * math.sqrt(self.horizon_days))

        # â”€â”€ Annualised vol â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        port_vol = float(sigma * math.sqrt(self.TRADING_DAYS))

        # â”€â”€ Limit check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        breached = h_var > self.var_limit or cvar > self.cvar_limit
        if h_var > self.var_limit:
            msg = f"VaR {h_var*100:.2f}% exceeds {self.var_limit*100:.1f}% limit â€” reduce exposure"
        elif cvar > self.cvar_limit:
            msg = f"CVaR {cvar*100:.2f}% exceeds {self.cvar_limit*100:.1f}% limit â€” tail risk elevated"
        else:
            msg = f"VaR {h_var*100:.2f}% within limits. CVaR {cvar*100:.2f}%."

        return VaRResult(
            confidence=self.confidence, horizon_days=self.horizon_days,
            historical_var=h_var, parametric_var=p_var, cvar=cvar,
            portfolio_vol=port_vol, limit_breached=breached,
            limit=self.var_limit, message=msg,
        )

    def apply_reduction(self, weights: dict[str, float]) -> dict[str, float]:
        """Scale all weights by reduction_factor on VaR breach."""
        return {t: w * self.reduction_factor for t, w in weights.items()}

    def _empty_result(self, msg: str = "No data") -> VaRResult:
        return VaRResult(0.95, 1, 0.0, 0.0, 0.0, 0.0, False, self.var_limit, msg)


# ===========================================================================
# 2. Drawdown Guard
# ===========================================================================

@dataclass
class DrawdownState:
    current_drawdown:    float      # peak-to-trough as fraction
    peak_value:          float
    current_value:       float
    days_in_drawdown:    int
    max_drawdown_30d:    float
    exposure_multiplier: float      # 1.0 = full, <1.0 = reduced
    circuit_breaker:     bool       # True = all new entries halted
    rule_triggered:      str        # description of which rule fired
    alerts:              list[str]

    def to_dict(self) -> dict:
        return {
            "current_drawdown_pct":  round(self.current_drawdown * 100, 2),
            "max_drawdown_30d_pct":  round(self.max_drawdown_30d * 100, 2),
            "days_in_drawdown":      self.days_in_drawdown,
            "exposure_multiplier":   round(self.exposure_multiplier, 2),
            "circuit_breaker":       self.circuit_breaker,
            "rule_triggered":        self.rule_triggered,
            "alerts":                self.alerts,
        }


class DrawdownGuard:
    """
    Monitors live portfolio equity curve and applies tiered exposure reduction.
    """

    def __init__(
        self,
        dd_soft:     float = 0.05,   # 5%  â†’ reduce to 80%
        dd_medium:   float = 0.10,   # 10% â†’ reduce to 50%
        dd_hard:     float = 0.15,   # 15% â†’ reduce to 25%
        dd_halt:     float = 0.20,   # 20% â†’ circuit breaker
        beta_limit:  float = 1.20,   # max portfolio beta vs Nifty
        turnover_limit: float = 0.10, # max 10% portfolio turnover/day
        theme_limit: float = 0.35,   # max 35% in any single theme
    ):
        self.dd_soft      = dd_soft
        self.dd_medium    = dd_medium
        self.dd_hard      = dd_hard
        self.dd_halt      = dd_halt
        self.beta_limit   = beta_limit
        self.turnover_limit = turnover_limit
        self.theme_limit  = theme_limit

    def evaluate(
        self,
        equity_curve: pd.Series,           # daily portfolio values
        nifty_returns: pd.Series | None = None,  # for beta calc
        current_positions: dict[str, Any] | None = None,
    ) -> DrawdownState:
        alerts: list[str] = []

        if equity_curve.empty or len(equity_curve) < 2:
            return DrawdownState(0, 0, 0, 0, 0, 1.0, False, "No data", [])

        # â”€â”€ Drawdown calculation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rolling_peak   = equity_curve.cummax()
        drawdown_series = (equity_curve - rolling_peak) / rolling_peak
        current_dd     = float(-drawdown_series.iloc[-1])
        max_dd_30d     = float(-drawdown_series.tail(30).min())
        peak_val       = float(rolling_peak.iloc[-1])
        current_val    = float(equity_curve.iloc[-1])

        # Days in current drawdown
        in_dd = (drawdown_series < -0.001)
        days_in_dd = int(in_dd[::-1].cumsum().iloc[-1]) if in_dd.any() else 0

        # â”€â”€ Tiered exposure rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        circuit_breaker = False
        rule = "No drawdown â€” full exposure"

        if current_dd >= self.dd_halt:
            multiplier      = 0.0
            circuit_breaker = True
            rule            = f"CIRCUIT BREAKER: Drawdown {current_dd*100:.1f}% â€” all new entries halted"
            alerts.append(rule)
        elif current_dd >= self.dd_hard:
            multiplier = 0.25
            rule       = f"HARD STOP: Drawdown {current_dd*100:.1f}% â€” exposure 25%, swing only"
            alerts.append(rule)
        elif current_dd >= self.dd_medium:
            multiplier = 0.50
            rule       = f"MEDIUM ALERT: Drawdown {current_dd*100:.1f}% â€” exposure 50%"
            alerts.append(rule)
        elif current_dd >= self.dd_soft:
            multiplier = 0.80
            rule       = f"SOFT ALERT: Drawdown {current_dd*100:.1f}% â€” exposure 80%"
            alerts.append(rule)
        else:
            multiplier = 1.0

        # â”€â”€ Beta check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if nifty_returns is not None and len(equity_curve) > 30:
            port_ret  = equity_curve.pct_change().dropna()
            common    = port_ret.index.intersection(nifty_returns.index)
            if len(common) > 20:
                p = port_ret.loc[common].values
                n = nifty_returns.loc[common].values
                cov_pn = float(np.cov(p, n)[0, 1])
                var_n  = float(np.var(n))
                beta   = cov_pn / var_n if var_n > 1e-10 else 1.0
                if beta > self.beta_limit:
                    alerts.append(f"Beta {beta:.2f} exceeds {self.beta_limit} limit â€” reduce high-beta names")
                    multiplier = min(multiplier, 0.70)

        return DrawdownState(
            current_drawdown=current_dd,
            peak_value=peak_val,
            current_value=current_val,
            days_in_drawdown=days_in_dd,
            max_drawdown_30d=max_dd_30d,
            exposure_multiplier=multiplier,
            circuit_breaker=circuit_breaker,
            rule_triggered=rule,
            alerts=alerts,
        )

    def check_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> tuple[float, bool]:
        """
        Compute portfolio turnover between old and new weights.
        Returns (turnover_fraction, is_within_limit).
        """
        all_tickers = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0) - old_weights.get(t, 0))
            for t in all_tickers
        ) / 2
        return round(turnover, 4), turnover <= self.turnover_limit

    def check_theme_concentration(
        self,
        weights: dict[str, float],
        themes: dict[str, str],  # {ticker: theme}
    ) -> list[str]:
        """Return list of themes exceeding the concentration limit."""
        theme_totals: dict[str, float] = {}
        for ticker, w in weights.items():
            t = themes.get(ticker, "Unknown")
            theme_totals[t] = theme_totals.get(t, 0) + w
        return [t for t, w in theme_totals.items() if w > self.theme_limit]


# ===========================================================================
# 3. Liquidity Filter
# ===========================================================================

@dataclass
class LiquidityCheck:
    ticker:              str
    avg_daily_vol_cr:    float        # average daily traded value â‚¹ crore
    position_size_cr:    float        # proposed position â‚¹ crore
    adv_pct:             float        # position as % of ADV
    days_to_liquidate:   float        # estimated days to exit at 10% ADV
    is_liquid:           bool
    recommended_size_cr: float        # adjusted size respecting ADV limit
    reason:              str


class LiquidityFilter:
    """
    Ensures no position exceeds a fraction of the stock's average daily volume.
    Illiquid positions cannot be exited cleanly in volatile markets.
    """

    def __init__(
        self,
        max_adv_pct:  float = 0.10,   # position â‰¤ 10% of ADV
        min_adv_cr:   float = 5.0,    # minimum â‚¹5 crore ADV to trade
        exit_days:    float = 3.0,    # flag if exit takes > 3 days
        adv_lookback: int   = 20,     # trading days for ADV calculation
    ):
        self.max_adv_pct  = max_adv_pct
        self.min_adv_cr   = min_adv_cr
        self.exit_days    = exit_days
        self.adv_lookback = adv_lookback

    def check(
        self,
        ticker: str,
        position_size_cr: float,
        volume_cr_series: pd.Series,  # daily traded value in â‚¹ crore
    ) -> LiquidityCheck:
        if volume_cr_series.empty:
            return LiquidityCheck(ticker, 0, position_size_cr, 0, 999,
                                  False, 0, "No volume data")

        adv = float(volume_cr_series.tail(self.adv_lookback).mean())
        adv_pct = position_size_cr / adv if adv > 0 else 999
        days_to_exit = position_size_cr / (adv * 0.10) if adv > 0 else 999

        too_large   = adv_pct > self.max_adv_pct
        too_illiquid = adv < self.min_adv_cr
        too_long_exit = days_to_exit > self.exit_days

        is_liquid = not (too_large or too_illiquid or too_long_exit)
        recommended = min(position_size_cr, adv * self.max_adv_pct)

        if too_illiquid:
            reason = f"ADV â‚¹{adv:.1f}cr below minimum â‚¹{self.min_adv_cr}cr â€” avoid"
        elif too_large:
            reason = f"Position {adv_pct*100:.1f}% of ADV â€” reduce to â‚¹{recommended:.1f}cr"
        elif too_long_exit:
            reason = f"Exit would take {days_to_exit:.1f} days â€” too illiquid"
        else:
            reason = f"Liquid. ADV â‚¹{adv:.1f}cr, position {adv_pct*100:.1f}% of ADV"

        return LiquidityCheck(
            ticker=ticker, avg_daily_vol_cr=round(adv, 2),
            position_size_cr=round(position_size_cr, 2),
            adv_pct=round(adv_pct, 4),
            days_to_liquidate=round(days_to_exit, 2),
            is_liquid=is_liquid,
            recommended_size_cr=round(recommended, 2),
            reason=reason,
        )

    def check_all(
        self,
        positions: dict[str, float],       # {ticker: capital_cr}
        volume_data: dict[str, pd.Series], # {ticker: daily_volume_cr_series}
    ) -> list[LiquidityCheck]:
        results = []
        for ticker, size_cr in positions.items():
            vol_series = volume_data.get(ticker, pd.Series(dtype=float))
            results.append(self.check(ticker, size_cr, vol_series))
        return sorted(results, key=lambda c: c.adv_pct, reverse=True)

    def apply_adjustments(
        self,
        weights: dict[str, float],
        checks: list[LiquidityCheck],
        total_capital: float,
    ) -> dict[str, float]:
        """Return adjusted weights respecting liquidity constraints."""
        adjusted = dict(weights)
        for check in checks:
            if not check.is_liquid and check.ticker in adjusted:
                old_size = adjusted[check.ticker] * total_capital
                new_size = check.recommended_size_cr
                if old_size > 0:
                    adjusted[check.ticker] *= (new_size / old_size)
        total = sum(adjusted.values())
        return {t: w / total for t, w in adjusted.items()} if total > 0 else adjusted

# ---------------------------------------------------------------------------
# 4. Correlation Controller
# ---------------------------------------------------------------------------
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
                        recommendation=f"Drop {t_b} â€” {rho:.2f} correlation with {t_a}.",
                    ))
                elif rho >= self.MODERATE_THRESHOLD:
                    flags.append(CorrelationFlag(
                        ticker_a=t_a, ticker_b=t_b,
                        correlation=round(rho, 3),
                        severity="MODERATE",
                        recommendation=f"Size both down â€” {rho:.2f} correlation.",
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

# Merged from engines/risk/advanced/risk_engine.py on consolidation
