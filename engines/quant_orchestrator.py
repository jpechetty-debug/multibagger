"""engines/quant_orchestrator.py
------------------------------
Single entry point that wires regime classification → cycle detection →
portfolio optimisation → Kelly sizing → advanced risk checks.

FIX-4: RELIANCE.NS proxy removed
---------------------------------
The previous version used RELIANCE.NS price history from universe_df as a
stand-in for the Nifty 50 index when computing regime volatility state.
RELIANCE tracks the index loosely but introduces single-stock idiosyncratic
noise that corrupts the VolatilityModel's ATR/realised-vol calculations.

Resolution: _fetch_nifty_prices() fetches ^NSEI from yfinance at runtime.
The result is cached as a class-level attribute so the expensive network
call is made at most once per process lifetime (refreshed if stale > 1hr).

Callers that pass nifty_prices explicitly in market_data["nifty_prices"]
take precedence — useful for backtesting where you supply historical prices
directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

import config
from engines.analysis.cycle_detector import CycleDetector
from engines.ml.performance_tracker import PerformanceTracker
from engines.portfolio.kelly_sizer import KellySizer
from engines.portfolio.optimizer import PortfolioConstraints, PortfolioOptimizer, PortfolioResult
from engines.regime.regime_engine import RegimeEngine, RegimeResult
from engines.risk.advanced_risk import AdvancedRiskEngine, DrawdownGuard, RiskCheckResult, VaREngine
from engines.risk.factor_audit import FactorAudit, FactorAuditReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nifty price cache
# ---------------------------------------------------------------------------

_NIFTY_CACHE: dict[str, Any] = {"prices": None, "fetched_at": 0.0}
_NIFTY_CACHE_TTL = 3600   # refresh at most once per hour


def _fetch_nifty_prices() -> pd.Series | None:
    """Fetch Nifty 50 close prices, using a 1-hour in-process cache.

    FIX-4: Returns the actual ^NSEI series instead of a single-stock proxy.
    Returns None on network failure so callers degrade gracefully.
    """
    now = time.time()
    if _NIFTY_CACHE["prices"] is not None and (now - _NIFTY_CACHE["fetched_at"]) < _NIFTY_CACHE_TTL:
        return _NIFTY_CACHE["prices"]

    try:
        import yfinance as yf
        hist = yf.Ticker(config.NIFTY_YFINANCE_SYMBOL).history(
            period=config.PRICE_HISTORY_PERIOD,
            interval=config.PRICE_HISTORY_INTERVAL,
            auto_adjust=True,
        )
        if hist.empty:
            logger.warning("QuantOrchestrator: Nifty price fetch returned empty — regime vol unavailable")
            return None
        prices = hist["Close"].dropna()
        _NIFTY_CACHE["prices"]     = prices
        _NIFTY_CACHE["fetched_at"] = now
        logger.debug("QuantOrchestrator: fetched %d Nifty price points", len(prices))
        return prices
    except Exception as exc:
        logger.warning("QuantOrchestrator: Nifty price fetch failed (%s) — regime vol unavailable", exc)
        return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QuantResult:
    regime:    RegimeResult
    portfolio: PortfolioResult
    risk:      RiskCheckResult
    var:       Any                # VaRResult
    drawdown:  Any                # DrawdownState
    audit:     FactorAuditReport
    timestamp: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "regime":    self.regime.to_payload(),
            "portfolio": self.portfolio.to_dict(),
            "risk":      self.risk.to_dict(),
            "var":       self.var.to_dict(),
            "drawdown":  self.drawdown.to_dict(),
            "audit":     self.audit.to_dict(),
        }

    def summary(self) -> str:
        r  = self.regime
        p  = self.portfolio
        rk = self.risk
        v  = self.var
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
    Coordinates engines in the correct sequence:
    Regime → Cycle → Filter → MVO → Kelly → Vol-scale → Risk → Record

    FIX-4: Nifty prices are fetched from ^NSEI, not proxied via RELIANCE.NS.
    Callers may still supply nifty_prices via market_data["nifty_prices"] for
    backtesting (explicit value always takes precedence over the live fetch).
    """

    def __init__(
        self,
        total_capital:         float,
        constraints:           PortfolioConstraints | None = None,
        max_correlation:       float = 0.75,
        target_daily_vol_pct:  float = 1.5,
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
        signals:           list[dict[str, Any]],
        market_data:       dict[str, Any],
        returns_df:        pd.DataFrame,
        universe_df:       pd.DataFrame,
        equity_curve:      pd.Series | None = None,
        benchmark_weights: dict[str, float] | None = None,
        prev_regime:       str | None = None,
        model_id:          str | None = None,
        signal_date:       str | None = None,
    ) -> QuantResult:

        # ── Step 1: Classify regime ────────────────────────────────────────
        # FIX-4: resolve Nifty prices — caller-supplied first, then live fetch
        nifty_prices: pd.Series | None = market_data.get("nifty_prices")
        if nifty_prices is None:
            nifty_prices = _fetch_nifty_prices()   # real ^NSEI, not RELIANCE proxy

        if nifty_prices is None:
            logger.warning("QuantOrchestrator: no Nifty prices available — regime vol signals degraded")

        regime = self.regime_engine.classify(
            market_data,
            previous_regime=prev_regime,
            nifty_prices=nifty_prices,
        )

        # ── Step 2: Filter signals by regime score threshold ───────────────
        eligible = [
            s for s in signals
            if s.get("total_score", 0) >= regime.score_threshold
            and s.get("action") in ("BUY", "WATCH")
        ]

        # ── Step 3: Correlation pre-filter ─────────────────────────────────
        corr_filtered, dropped_by_corr = self.risk_engine.cc.filter_candidates(
            eligible, returns_df
        )

        # ── Step 4: MVO portfolio optimisation ─────────────────────────────
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

        # ── Step 4.5: Kelly sizing overlay ─────────────────────────────────
        raw_weights = {p.ticker: p.weight for p in portfolio.positions}
        kelly_res   = self.kelly_sizer.adjust(
            weights=raw_weights,
            signal_records=corr_filtered,
            tracker=self.tracker,
            model_id=model_id,
        )
        kelly_weights = kelly_res.weights()
        for pos in portfolio.positions:
            if pos.ticker in kelly_weights:
                pos.weight            = kelly_weights[pos.ticker]
                pos.position_pct      = pos.weight * 100
                pos.capital_allocated = portfolio.effective_capital * pos.weight
        if kelly_res.cash_implied > 0.15:
            portfolio.warnings.append(kelly_res.summary())

        # ── Step 4.6: Cycle-aware sector weight adjustment ─────────────────
        cycle_result = self.cycle_detector.load_latest()
        if cycle_result is not None:
            max_pos = self._constraints.max_position_pct
            for pos in portfolio.positions:
                sig    = next((s for s in corr_filtered if s.get("ticker") == pos.ticker), {})
                sector = sig.get("sector", "")
                if sector:
                    mult  = cycle_result.sector_multiplier(sector)
                    new_w = min(pos.weight * mult, max_pos)
                    pos.weight            = new_w
                    pos.position_pct      = new_w * 100
                    pos.capital_allocated = portfolio.effective_capital * new_w

            # Re-normalise
            total_w = sum(p.weight for p in portfolio.positions)
            if total_w > 0:
                for pos in portfolio.positions:
                    pos.weight            = round(pos.weight / total_w * min(total_w, 1.0), 4)
                    pos.position_pct      = pos.weight * 100
                    pos.capital_allocated = portfolio.effective_capital * pos.weight

            portfolio.warnings.append(
                f"Cycle: {cycle_result.phase} (conf {cycle_result.confidence:.0%}) "
                f"bias {cycle_result.position_bias:.2f}x"
            )

        # ── Step 5: Volatility scaling ──────────────────────────────────────
        raw_weights = {p.ticker: p.weight for p in portfolio.positions}
        vol_scaled  = self.risk_engine.vs.scale(
            [p.ticker for p in portfolio.positions], raw_weights, returns_df
        )
        scaled_map = {p.ticker: p.scaled_weight for p in vol_scaled}
        for pos in portfolio.positions:
            if pos.ticker in scaled_map:
                pos.weight            = scaled_map[pos.ticker]
                pos.position_pct      = pos.weight * 100
                pos.capital_allocated = portfolio.effective_capital * pos.weight

        # ── Step 6: Advanced risk ───────────────────────────────────────────
        final_weights = {p.ticker: p.weight for p in portfolio.positions}
        var_res       = self.var_engine.compute(final_weights, returns_df)

        if equity_curve is None:
            equity_curve = pd.Series(
                [self.total_capital, self.total_capital], index=[0, 1]
            )
        dd_res = self.dd_guard.evaluate(equity_curve)

        factor_scores = pd.DataFrame([
            {"ticker": t, "value": 0, "quality": 1.0, "momentum": 0.5, "size": -0.2}
            for t in final_weights
        ])
        audit_res = self.factor_audit.run(final_weights, factor_scores, benchmark_weights)

        risk = self.risk_engine.check(
            candidates=corr_filtered,
            weights=final_weights,
            returns_df=returns_df,
            universe_df=universe_df,
        )
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

        # ── Step 7: Record predictions for performance tracking ────────────
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
