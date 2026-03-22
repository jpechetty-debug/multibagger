"""
engines/portfolio/optimizer.py
--------------------------------
Regime-aware portfolio optimizer for NSE equities.

Approach
--------
Mean-variance optimization (Markowitz) with practical NSE constraints:
  - Maximum position size per stock
  - Maximum sector concentration
  - Minimum position size (avoids hairline allocations)
  - Correlation penalty to avoid concentrated bets
  - Regime-aware position multiplier applied to all sizes

Optimization objective
----------------------
    maximise: w' · μ - λ · w' · Σ · w

Where:
    w  = weight vector
    μ  = expected return proxy (normalised signal scores)
    Σ  = covariance matrix of returns
    λ  = risk aversion parameter (regime-adjusted)
    subject to: sum(w) = 1, w_i >= min_weight, w_i <= max_weight,
                sector(s) <= max_sector_weight
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Constraints (overridable per portfolio)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConstraints:
    max_position_pct:     float = 0.08    # max 8% per stock
    min_position_pct:     float = 0.02    # min 2% (no hairline allocations)
    max_sector_pct:       float = 0.30    # max 30% per sector
    max_positions:        int   = 20      # maximum number of stocks
    min_positions:        int   = 8       # minimum for diversification
    risk_aversion:        float = 2.0     # λ — higher = more conservative
    correlation_penalty:  float = 0.5     # extra penalty for correlated pairs


REGIME_RISK_AVERSION = {
    "BULL":     1.5,
    "QUALITY":  2.0,
    "SIDEWAYS": 3.0,
    "BEAR":     5.0,
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PositionAllocation:
    ticker: str
    sector: str
    signal_score: float
    weight: float           # optimal portfolio weight 0–1
    capital_allocated: float  # ₹ amount
    shares_approx: int      # approximate share count
    position_pct: float     # weight as percentage


@dataclass
class PortfolioResult:
    total_capital: float
    regime: str
    regime_multiplier: float
    effective_capital: float       # total_capital × regime_multiplier
    positions: list[PositionAllocation]
    expected_return_proxy: float   # weighted score proxy
    portfolio_volatility: float    # annualised, estimated
    sharpe_proxy: float            # expected_return / volatility
    sector_weights: dict[str, float]
    excluded: list[str]            # tickers dropped (constraints / correlation)
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "regime":              self.regime,
            "regime_multiplier":   self.regime_multiplier,
            "effective_capital":   round(self.effective_capital, 0),
            "position_count":      len(self.positions),
            "expected_return_proxy": round(self.expected_return_proxy, 2),
            "portfolio_volatility":  round(self.portfolio_volatility * 100, 2),
            "sharpe_proxy":        round(self.sharpe_proxy, 3),
            "sector_weights":      {k: round(v, 3) for k, v in self.sector_weights.items()},
            "excluded_count":      len(self.excluded),
            "warnings":            self.warnings,
            "positions": [
                {
                    "ticker":           p.ticker,
                    "sector":           p.sector,
                    "signal_score":     round(p.signal_score, 1),
                    "weight":           round(p.weight, 4),
                    "position_pct":     round(p.position_pct, 2),
                    "capital_allocated": round(p.capital_allocated, 0),
                }
                for p in sorted(self.positions, key=lambda x: x.weight, reverse=True)
            ],
        }


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Regime-aware mean-variance portfolio optimizer.

    Usage
    -----
        optimizer = PortfolioOptimizer(total_capital=1_000_000)
        result = optimizer.optimize(
            candidates=signal_records,   # list of dicts with ticker, sector, score, price
            returns_df=returns_df,        # DataFrame of daily returns (tickers × dates)
            regime="BULL",
            regime_multiplier=1.0,
        )
    """

    def __init__(
        self,
        total_capital: float,
        constraints: PortfolioConstraints | None = None,
    ):
        self.total_capital = total_capital
        self.c = constraints or PortfolioConstraints()

    def optimize(
        self,
        candidates: list[dict[str, Any]],
        returns_df: pd.DataFrame,
        regime: str = "QUALITY",
        regime_multiplier: float = 1.0,
    ) -> PortfolioResult:
        warnings_list: list[str] = []

        # ── Risk aversion from regime ─────────────────────────────────────────
        lam = REGIME_RISK_AVERSION.get(regime, self.c.risk_aversion)

        # ── Filter to candidates with price data ─────────────────────────────
        tickers_with_data = [
            c for c in candidates
            if c.get("ticker") in returns_df.columns
            and c.get("price", 0) > 0
            and c.get("total_score", 0) > 0
        ]
        if len(tickers_with_data) < self.c.min_positions:
            warnings_list.append(
                f"Only {len(tickers_with_data)} candidates have price data — "
                f"minimum {self.c.min_positions} required."
            )

        # ── Take top N by score ───────────────────────────────────────────────
        tickers_with_data.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        pool = tickers_with_data[: self.c.max_positions * 2]  # 2× budget before correlation filter

        if not pool:
            return self._empty_result(regime, regime_multiplier, ["No valid candidates"])

        tickers = [c["ticker"] for c in pool]
        sectors = {c["ticker"]: c.get("sector", "Unknown") for c in pool}
        prices  = {c["ticker"]: c.get("price", 100) for c in pool}
        scores  = np.array([c.get("total_score", 50) for c in pool])

        # ── Covariance matrix from returns ────────────────────────────────────
        ret = returns_df[tickers].dropna()
        if len(ret) < 30:
            warnings_list.append("Fewer than 30 days of return history — covariance estimate may be unstable.")
        cov = ret.cov().values * 252    # annualised
        cov += np.eye(len(tickers)) * 1e-6  # regularise

        # ── Normalise expected return proxy ───────────────────────────────────
        mu = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        # ── Gradient-free optimisation via projected gradient ─────────────────
        weights = self._optimise_weights(mu, cov, lam, len(tickers))

        # ── Apply sector concentration constraint ─────────────────────────────
        weights, excluded_by_sector = self._apply_sector_constraint(
            weights, tickers, sectors
        )

        # ── Drop below minimum weight ─────────────────────────────────────────
        excluded_small: list[str] = []
        for i, w in enumerate(weights):
            if 0 < w < self.c.min_position_pct:
                excluded_small.append(tickers[i])
                weights[i] = 0.0

        # Renormalise
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w

        # ── Build positions ───────────────────────────────────────────────────
        effective_capital = self.total_capital * regime_multiplier
        positions: list[PositionAllocation] = []
        for i, ticker in enumerate(tickers):
            if weights[i] < self.c.min_position_pct:
                continue
            capital = effective_capital * weights[i]
            price   = prices[ticker]
            positions.append(PositionAllocation(
                ticker=ticker,
                sector=sectors[ticker],
                signal_score=scores[i],
                weight=weights[i],
                capital_allocated=capital,
                shares_approx=max(1, int(capital / price)),
                position_pct=weights[i] * 100,
            ))

        # ── Portfolio metrics ─────────────────────────────────────────────────
        active_weights = np.array([weights[i] for i, t in enumerate(tickers)
                                   if weights[i] >= self.c.min_position_pct])
        active_idx     = [i for i, t in enumerate(tickers)
                          if weights[i] >= self.c.min_position_pct]
        if len(active_idx) >= 2:
            active_cov   = cov[np.ix_(active_idx, active_idx)]
            port_var     = active_weights @ active_cov @ active_weights
            port_vol     = float(np.sqrt(max(port_var, 0)))
            exp_ret      = float(active_weights @ mu[active_idx])
            sharpe_proxy = exp_ret / (port_vol + 1e-9)
        else:
            port_vol, exp_ret, sharpe_proxy = 0.0, 0.0, 0.0

        # ── Sector weights ────────────────────────────────────────────────────
        sector_weights: dict[str, float] = {}
        for p in positions:
            sector_weights[p.sector] = sector_weights.get(p.sector, 0) + p.weight

        all_excluded = list(set(excluded_by_sector + excluded_small))

        return PortfolioResult(
            total_capital=self.total_capital,
            regime=regime,
            regime_multiplier=regime_multiplier,
            effective_capital=effective_capital,
            positions=positions,
            expected_return_proxy=exp_ret,
            portfolio_volatility=port_vol,
            sharpe_proxy=sharpe_proxy,
            sector_weights=sector_weights,
            excluded=all_excluded,
            warnings=warnings_list,
        )

    def _optimise_weights(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        lam: float,
        n: int,
        iterations: int = 500,
        lr: float = 0.01,
    ) -> np.ndarray:
        """
        Projected gradient ascent on the MV objective.
        Constraints: sum=1, w_i in [0, max_position_pct].
        """
        w = np.ones(n) / n
        for _ in range(iterations):
            grad = mu - lam * (cov @ w)
            w    = w + lr * grad
            w    = np.clip(w, 0.0, self.c.max_position_pct)
            s    = w.sum()
            if s > 0:
                w = w / s
        return w

    def _apply_sector_constraint(
        self,
        weights: np.ndarray,
        tickers: list[str],
        sectors: dict[str, str],
    ) -> tuple[np.ndarray, list[str]]:
        excluded: list[str] = []
        for _ in range(50):
            sector_totals: dict[str, float] = {}
            for i, ticker in enumerate(tickers):
                sec = sectors[ticker]
                sector_totals[sec] = sector_totals.get(sec, 0) + weights[i]

            over = {s: t for s, t in sector_totals.items() if t > self.c.max_sector_pct}
            if not over:
                break

            for sector, total in over.items():
                excess = total - self.c.max_sector_pct
                sector_positions = [
                    (i, weights[i]) for i, t in enumerate(tickers)
                    if sectors[t] == sector and weights[i] > 0
                ]
                if not sector_positions:
                    break
                worst_idx = max(sector_positions, key=lambda x: x[1])[0]
                cut = min(excess, weights[worst_idx])
                weights[worst_idx] -= cut
                if weights[worst_idx] < self.c.min_position_pct:
                    excluded.append(tickers[worst_idx])
                    weights[worst_idx] = 0.0

            total_w = weights.sum()
            if total_w > 0:
                weights = weights / total_w

        return weights, excluded

    def _empty_result(self, regime: str, multiplier: float, warnings: list[str]) -> PortfolioResult:
        return PortfolioResult(
            total_capital=self.total_capital,
            regime=regime,
            regime_multiplier=multiplier,
            effective_capital=self.total_capital * multiplier,
            positions=[], expected_return_proxy=0.0,
            portfolio_volatility=0.0, sharpe_proxy=0.0,
            sector_weights={}, excluded=[], warnings=warnings,
        )
