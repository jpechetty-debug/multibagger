"""
engines/quant_orchestrator.py
------------------------------
Single entry point that wires regime classification → cycle detection →
portfolio optimisation → Kelly sizing → advanced risk checks.

Triple-gate exposure control
-----------------------------
1. Regime multiplier   — position_multiplier from 8-state RegimeTrackerV2
2. Kelly sizer         — fractional Kelly preserves cash when edge is low
3. Volatility scaling  — inverse-vol sizing equalises rupee risk per position

Cycle-aware sector weights
--------------------------
CycleDetector runs daily (via scheduler) and provides sector_multiplier()
for each NSE sector.  QuantOrchestrator applies this as a fourth layer on
top of the MVO weights — leaders get 1.2×, laggards get 0.6×, neutral 1.0×.
All multipliers are capped so no single position exceeds max_position_pct.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from engines.regime.regime_engine import RegimeEngine, RegimeResult
from engines.portfolio.optimizer import PortfolioOptimizer, PortfolioConstraints, PortfolioResult
from engines.risk.advanced_risk import AdvancedRiskEngine, RiskCheckResult
from engines.risk.advanced_risk import VaREngine, DrawdownGuard
from engines.risk.factor_audit import FactorAudit, FactorAuditReport
from engines.ml.performance_tracker import PerformanceTracker
from engines.portfolio.kelly_sizer import KellySizer
from engines.analysis.cycle_detector import CycleDetector


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QuantResult:
    regime:       RegimeResult
    portfolio:    PortfolioResult
    risk:         RiskCheckResult
    var:          Any                # VaRResult
    drawdown:     Any                # DrawdownState
    audit:        FactorAuditReport
    timestamp:    int

    def to_dict(self) -> dict:
        return {
            "timestamp":  self.timestamp,
            "regime":     self.regime.to_payload(),
            "portfolio":  self.portfolio.to_dict(),
            "risk":       self.risk.to_dict(),
            "var":        self.var.to_dict(),
            "drawdown":   self.drawdown.to_dict(),
            "audit":      self.audit.to_dict(),
        }

    def summary(self) -> str:
        r = self.regime
        p = self.portfolio
        rk = self.risk
        v = self.var
        return (
            f"Regime: {r.regime} ({r.composite_score:.0f}/100) | "
            f"Portfolio: {len(p.positions)} positions | "
            f"VaR (95%): {v.historical_var*100:.2f}% | "
            f"Risk: {'ACTION REQUIRED' if rk.action_required or v.limit_breached else 'OK'}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class QuantOrchestrator:
    """
    Coordinates the engines in the correct sequence:
    Regime → Cycle → Filter → MVO → Kelly → Vol-scale → Risk → Record

    CycleDetector integration
    -------------------------
    Loads the latest daily cycle snapshot (saved by scheduler at 07:00 IST).
    Applies sector_multiplier() to each position's weight after MVO + Kelly.
    Leaders (e.g. Financials in EARLY cycle) get up to 1.2× boost.
    Laggards (e.g. FMCG in EARLY cycle) get 0.6× reduction.
    All final weights are re-capped to max_position_pct after adjustment.
    Falls back gracefully if no cycle snapshot exists yet.
    """

    def __init__(
        self,
        total_capital: float,
        constraints: PortfolioConstraints | None = None,
        max_correlation: float = 0.75,
        target_daily_vol_pct: float = 1.5,
    ):
        self.total_capital  = total_capital
        self.regime_engine  = RegimeEngine()
        self.risk_engine    = AdvancedRiskEngine(
            max_correlation=max_correlation,
            target_daily_vol_pct=target_daily_vol_pct,
        )
        self.var_engine     = VaREngine()
        self.dd_guard       = DrawdownGuard()
        self.factor_audit   = FactorAudit()
        self._constraints   = constraints or PortfolioConstraints()
        self.kelly_sizer    = KellySizer(
            kelly_fraction=0.25,
            max_weight=self._constraints.max_position_pct,
        )
        self.tracker        = PerformanceTracker()
        self.cycle_detector = CycleDetector()
        self.holding_days   = 20

    def run(
        self,
        signals: list[dict[str, Any]],
        market_data: dict[str, Any],
        returns_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        equity_curve: pd.Series | None = None,
        benchmark_weights: dict[str, float] | None = None,
        prev_regime: str | None = None,
        model_id: str | None = None,
        signal_date: str | None = None,
    ) -> QuantResult:

        # ── Step 1: Classify regime ───────────────────────────────────────────
        # Pass nifty_prices if available in universe_df
        nifty_prices = universe_df["RELIANCE.NS"] if "RELIANCE.NS" in universe_df.columns else None # proxy
        regime = self.regime_engine.classify(market_data, previous_regime=prev_regime, nifty_prices=nifty_prices)

        # ── Step 2: Filter signals by regime score threshold ──────────────────
        eligible = [
            s for s in signals
            if s.get("total_score", 0) >= regime.score_threshold
            and s.get("action") in ("BUY", "WATCH")
        ]

        # ── Step 3: Correlation pre-filter ────────────────────────────────────
        corr_filtered, dropped_by_corr = self.risk_engine.cc.filter_candidates(
            eligible, returns_df
        )

        # ── Step 4: Optimise portfolio ────────────────────────────────────────
        optimizer = PortfolioOptimizer(
            total_capital=self.total_capital,
            constraints=self._constraints,
        )
        portfolio = optimizer.optimize(
            candidates=corr_filtered,
            returns_df=returns_df,
            regime=regime.regime,
            regime_multiplier=regime.position_multiplier,
        )

        # ── Step 4.5: Kelly position sizing overlay ───────────────────────────
        raw_weights = {p.ticker: p.weight for p in portfolio.positions}
        kelly_res = self.kelly_sizer.adjust(
            weights=raw_weights,
            signal_records=corr_filtered,
            tracker=self.tracker,
            model_id=model_id,
        )
        kelly_weights = kelly_res.weights()

        for pos in portfolio.positions:
            if pos.ticker in kelly_weights:
                pos.weight = kelly_weights[pos.ticker]
                pos.position_pct = pos.weight * 100
                pos.capital_allocated = portfolio.effective_capital * pos.weight

        if kelly_res.cash_implied > 0.15:
            portfolio.warnings.append(kelly_res.summary())

        # ── Step 4.6: Cycle-aware sector weight adjustment ────────────────────
        # Load the latest daily cycle snapshot (saved by scheduler at 07:00 IST).
        # Adjusts MVO+Kelly weights up/down based on sector cycle phase.
        # Falls back gracefully if snapshot is missing.
        cycle_result = self.cycle_detector.load_latest()
        if cycle_result is not None:
            max_pos = self._constraints.max_position_pct
            for pos in portfolio.positions:
                # Look up sector from signal records
                sig = next((s for s in corr_filtered if s.get("ticker") == pos.ticker), {})
                sector = sig.get("sector", "")
                if sector:
                    mult = cycle_result.sector_multiplier(sector)
                    new_w = min(pos.weight * mult, max_pos)
                    pos.weight = new_w
                    pos.position_pct = new_w * 100
                    pos.capital_allocated = portfolio.effective_capital * new_w
            # Re-normalise so weights still reflect intended deployment fraction
            total_w = sum(p.weight for p in portfolio.positions)
            if total_w > 0:
                for pos in portfolio.positions:
                    pos.weight = round(pos.weight / total_w * min(total_w, 1.0), 4)
                    pos.position_pct = pos.weight * 100
                    pos.capital_allocated = portfolio.effective_capital * pos.weight
            portfolio.warnings.append(
                f"Cycle: {cycle_result.phase} (conf {cycle_result.confidence:.0%}) "
                f"bias {cycle_result.position_bias:.2f}x"
            )

        # ── Step 5: Volatility-scale the optimised weights ────────────────────
        raw_weights = {p.ticker: p.weight for p in portfolio.positions}
        vol_scaled  = self.risk_engine.vs.scale(
            [p.ticker for p in portfolio.positions], raw_weights, returns_df
        )
        
        # Update portfolio positions with vol-scaled weights
        scaled_map = {p.ticker: p.scaled_weight for p in vol_scaled}
        for pos in portfolio.positions:
            if pos.ticker in scaled_map:
                pos.weight       = scaled_map[pos.ticker]
                pos.position_pct = pos.weight * 100
                pos.capital_allocated = portfolio.effective_capital * pos.weight

        # ── Step 6: Advanced Risk (VaR & Drawdown) ────────────────────────────
        final_weights = {p.ticker: p.weight for p in portfolio.positions}
        
        # VaR
        var_res = self.var_engine.compute(final_weights, returns_df)
        
        # Drawdown Guard
        if equity_curve is None:
            # Mock equity curve if not provided (for first run)
            equity_curve = pd.Series([self.total_capital, self.total_capital], index=[0, 1])
        dd_res = self.dd_guard.evaluate(equity_curve)

        # Factor Audit
        # Mock factor scores if not provided (needs integration with feature store)
        factor_scores = pd.DataFrame([{"ticker": t, "value": 0, "quality": 1.0, "momentum": 0.5, "size": -0.2} for t in final_weights])
        audit_res = self.factor_audit.run(final_weights, factor_scores, benchmark_weights)

        # Legacy risk check
        risk = self.risk_engine.check(
            candidates=corr_filtered,
            weights=final_weights,
            returns_df=returns_df,
            universe_df=universe_df,
        )

        # Append correlation-dropped tickers to portfolio excluded list
        portfolio.excluded.extend(dropped_by_corr)

        res = QuantResult(
            regime=regime,
            portfolio=portfolio,
            risk=risk,
            var=var_res,
            drawdown=dd_res,
            audit=audit_res,
            timestamp=int(time.time()),
        )

        # ── Step 7: Record predictions for tracking ───────────────────────────
        if model_id and signal_date:
            records = [
                {
                    "model_id":         model_id,
                    "ticker":           pos.ticker,
                    "signal_date":      signal_date,
                    "predicted_score":  pos.signal_score,
                    "predicted_return": pos.signal_score / 1000,
                    "holding_days":     self.holding_days,
                }
                for pos in portfolio.positions
                if pos.signal_score > 0
            ]
            if records:
                self.tracker.record_batch_predictions(records)

        return res
